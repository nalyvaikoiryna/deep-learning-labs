import argparse
import json
from pathlib import Path
import platform
import subprocess
import torch
import pandas as pd
import torch.nn as nn
from torchvision.models import resnet18
import onnx
from PIL import Image
from torchvision import transforms
import onnxruntime as ort
import numpy as np
from sklearn.metrics import f1_score
import time
from onnxruntime.quantization import (
    CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
    quant_pre_process, quantize_static,
)
from onnx import numpy_helper

# Підготовка: перевірка поділу з ЛР3
# з ЛР3


def check_split(csv_path):
    table = pd.read_csv(csv_path)
    if set(table.columns) != {"file", "label", "part"}:
        raise ValueError("Стовпці CSV мають бути file,label,part")
    if table.isna().any().any():
        raise ValueError("У split.csv є порожні значення")
    if table["file"].duplicated().any():
        raise ValueError("У split.csv є повтори файлів, зокрема можливі перетини частин")
    if not table["file"].map(lambda name: Path(name).name == name).all():
        raise ValueError("У стовпці file потрібні імена без шляхів")

    expected = {
        "buildings": [(200, 1996761), (218, 4837323), (219, 4837559)],
        "forest": [(200, 2032625), (237, 5276962), (237, 5291326)],
        "glacier": [(200, 2183923), (276, 6135652), (277, 6157250)],
        "mountain": [(200, 1950345), (262, 5789535), (263, 5849765)],
        "sea": [(200, 1987613), (255, 5655068), (255, 5668011)],
        "street": [(200, 1841817), (250, 5541845), (251, 5547533)],
    }
    parts = ["train", "validation", "test"]
    if set(table["label"]) != set(expected) or set(table["part"]) != set(parts):
        raise ValueError("Неправильні назви класів або частин у split.csv")
    table["file_id"] = table["file"].map(lambda name: int(Path(name).stem))
    summary = table.groupby(["label", "part"]).agg(
        count=("file_id", "size"), file_sum=("file_id", "sum")
    )
    index = pd.MultiIndex.from_product([list(expected), parts], names=["label", "part"])
    summary = summary.reindex(index).reset_index()
    pairs = [pair for values in expected.values() for pair in values]
    summary["expected_count"] = [pair[0] for pair in pairs]
    summary["expected_sum"] = [pair[1] for pair in pairs]
    summary["matches_expected"] = (
        (summary["count"] == summary["expected_count"])
        & (summary["file_sum"] == summary["expected_sum"])
    )
    print(summary.to_string(index=False))
    if not summary["matches_expected"].all():
        raise ValueError("Поділ не збігається з контрольною таблицею")
    return summary

# Пункт 1: відновлення й експорт моделі FP32
def load_lab3_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    expected_classes = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
    if list(checkpoint["classes"]) != expected_classes:
        raise ValueError("Порядок класів у моделі не збігається з ЛР3")
    if checkpoint["img_size"] != 160:
        raise ValueError("Розмір входу з чекпойнта має дорівнювати 160")
    model = resnet18(weights=None)
    model.fc = nn.Linear(512, 6)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint

def export_fp32(model, img_size, output_path):
    example = torch.zeros((1, 3, img_size, img_size))  # Вхідний тензор для експорту

    torch.onnx.export(
        model,
        example,
        output_path,
        dynamo=False,
        opset_version=17,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
                "input": {0: "batch"},
                "logits": {0: "batch"},
            },
    )
def load_test_batch(batch_rows, data_dir, preprocess, class_to_idx):
    images = []
    labels = []

    for row in batch_rows.itertuples(index=False):
        path = data_dir / "seg_test" / "seg_test" / row.label / row.file
        with Image.open(path) as image:
            images.append(preprocess(image.convert("RGB")))
        labels.append(class_to_idx[row.label])

    return torch.stack(images), labels


# Пункт 2.2: подача калібрувальних зображень по одному
class ImageCalibrationReader(CalibrationDataReader):
    def __init__(self, rows, data_dir, preprocess):
        self.rows = iter(rows.itertuples(index=False))
        self.data_dir = data_dir
        self.preprocess = preprocess

    def get_next(self):
        row = next(self.rows, None)
        if row is None:
            return None

        image_path = self.data_dir / "seg_train" / "seg_train" / row.label / row.file
        with Image.open(image_path) as image:
            x = self.preprocess(image.convert("RGB")).unsqueeze(0).numpy()

        return {"input": x}


# Пункт 4: розкид діапазонів ваг між вихідними каналами згорток
def conv_weight_range_ratios(graph):
    initializers = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in graph.initializer
    }
    ratios = []
    for node in graph.node:
        if node.op_type != "Conv":
            continue
        weights = initializers[node.input[1]]
        if weights.ndim != 4:
            raise ValueError(f"Неочікувана форма ваг згортки {node.name}: {weights.shape}")
        channel_ranges = np.max(np.abs(weights), axis=(1, 2, 3))
        median_range = np.median(channel_ranges)
        if median_range <= 0:
            raise ValueError(f"Нульова медіана діапазонів у згортці {node.name}")
        r = float(channel_ranges.max() / median_range)
        ratios.append((node.name, r))
    return sorted(ratios, key=lambda item: item[1], reverse=True)


# Пункт 5: затримка одного виклику ONNX Runtime без підготовки входу
def benchmark_sessions(session_fp32, session_int8, input_array, rounds=3,
                       warmup_calls=20, measured_calls=100):
    if input_array.shape != (1, 3, 160, 160) or input_array.dtype != np.float32:
        raise ValueError("Для затримки потрібен float32-вхід форми (1, 3, 160, 160)")

    output_names = ["logits"]
    input_feed = {"input": input_array}
    sessions = (("FP32", session_fp32), ("INT8", session_int8))
    samples_ms = {name: [] for name, _ in sessions}
    round_p50_ms = {name: [] for name, _ in sessions}

    for _ in range(rounds):
        for name, sess in sessions:  # у кожному раунді спершу FP32, потім INT8
            for _ in range(warmup_calls):
                sess.run(output_names, input_feed)

            round_samples = []
            for _ in range(measured_calls):
                start = time.perf_counter()
                sess.run(output_names, input_feed)
                end = time.perf_counter()
                round_samples.append((end - start) * 1000)

            samples_ms[name].extend(round_samples)
            round_p50_ms[name].append(float(np.median(round_samples)))

    return {
        name: {
            "samples_ms": samples_ms[name],
            "round_p50_ms": round_p50_ms[name],
            "p50_ms": float(np.percentile(samples_ms[name], 50)),
            "p95_ms": float(np.percentile(samples_ms[name], 95)),
        }
        for name, _ in sessions
    }


def cpu_model_name():
    if platform.system() == "Darwin":
        try:
            hardware = subprocess.run(
                ["system_profiler", "SPHardwareDataType"],
                capture_output=True, text=True, check=True, timeout=30,
            ).stdout
            for line in hardware.splitlines():
                label, _, value = line.strip().partition(":")
                if label in {"Chip", "Processor Name"} and value.strip():
                    return value.strip()
        except (FileNotFoundError, subprocess.CalledProcessError,
                subprocess.TimeoutExpired):
            pass
    return platform.processor() or platform.machine()


def runtime_environment(options):
    os_name = (
        f"macOS {platform.mac_ver()[0]}"
        if platform.system() == "Darwin" else platform.platform()
    )
    return {
        "cpu": cpu_model_name(),
        "os": os_name,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": ort.__version__,
        "intra_op_num_threads": options.intra_op_num_threads,
        "inter_op_num_threads": options.inter_op_num_threads,
        "batch_size": 1,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="ЛР4: FP32 та INT8 для ResNet18")
    parser.add_argument("--model", type=Path, default=Path("../lab3/lab3_model.pt"))
    parser.add_argument("--split", type=Path, default=Path("../lab3/split.csv"))
    parser.add_argument("--data-dir", type=Path, default=Path("../lab3"))
    parser.add_argument("--results", type=Path, default=Path("lab4_results.json"))
    return parser.parse_args()


def write_results_json(path, results):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
        file.write("\n")


def main(args):
    # Підготовка: контрольні числа split.csv
    summary = check_split(args.split)
    totals = summary.groupby("part")[["count", "file_sum"]].sum()
    print(totals.loc[["train", "test"]])

    # Пункт 1.1: модель із ЛР3
    model, checkpoint = load_lab3_model(args.model)
    print(checkpoint["img_size"])  # очікуємо 160
    print(model.training)         # очікуємо False


    # Пункт 1.2: експорт FP32, час і вузли графа
    start = time.perf_counter() # додала час, коли дійшла до відповідного пункту 
    export_fp32(model, checkpoint["img_size"], "lab4_fp32.onnx")
    export_seconds = time.perf_counter() - start
    print(f"Час експорту: {export_seconds:.3f} с")
    onnx.checker.check_model("lab4_fp32.onnx")
    graph = onnx.load("lab4_fp32.onnx").graph
    node_types = [node.op_type for node in graph.node]
    print(f"Кількість вузлів Conv: {node_types.count('Conv')}")
    print(f"Кількість вузлів BatchNormalization: {node_types.count('BatchNormalization')}")
    

    # Пункт 1.3: тестові зображення й перетворення входу
    split = pd.read_csv(args.split)
    data_dir = args.data_dir
    img_size = checkpoint["img_size"]
    preprocess = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    # Пункт 1.4: порівняння PyTorch та ONNX на всьому test
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        "lab4_fp32.onnx",
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    test_rows = split.loc[split["part"] == "test"]
    class_to_idx = {name: i for i, name in enumerate(checkpoint["classes"])}
    pytorch_batches = []
    onnx_batches = []
    y_true = []

    for start in range(0, len(test_rows), 64):
        batch_rows = test_rows.iloc[start:start + 64]
        x, labels = load_test_batch(batch_rows, data_dir, preprocess, class_to_idx)
        with torch.no_grad():
            logits_tensor = model(x)
        logits_pytorch = logits_tensor.numpy()
        pytorch_batches.append(logits_pytorch)
        logits_onnx = session.run(["logits"], {"input": x.numpy()})[0]
        onnx_batches.append(logits_onnx)
        y_true.extend(labels)

    # Пункт 1.5: різниця логітів, збіг класів і macro-F1
    logits_pytorch = np.concatenate(pytorch_batches, axis=0)
    logits_onnx = np.concatenate(onnx_batches, axis=0)
    diff = np.abs(logits_pytorch - logits_onnx).max()
    print(f"Максимальна різниця між PyTorch та ONNX: {diff:.6f}")
    max_logit_pytorch = logits_pytorch.argmax(axis=1)
    max_logit_onnx = logits_onnx.argmax(axis=1)
    agreement = np.mean(max_logit_pytorch == max_logit_onnx)
    print(f"Частка збігів передбаченого класу: {agreement:.6f}")
    f1_pytorch = f1_score(
        y_true, max_logit_pytorch,
        labels=range(6), average="macro", zero_division=0,
    )
    f1_onnx = f1_score(
        y_true, max_logit_onnx,
        labels=range(6), average="macro", zero_division=0,
    )
    print(f"F1 PyTorch: {f1_pytorch:.6f}")
    print(f"F1 ONNX: {f1_onnx:.6f}")

    # Пункт 2.1: початкова перевірка train для калібрування
    train_rows = split.loc[split["part"] == "train"].copy()
    print(train_rows.groupby("label").size())
    train_rows = train_rows.sort_values("file").reset_index(drop=True)
    idx = np.random.default_rng(0).permutation(1200)[:200]
    calibration_rows = train_rows.iloc[idx]
    print(len(calibration_rows))
    print(calibration_rows["file"].map(lambda name: int(Path(name).stem)).sum())
    print(calibration_rows.groupby("label").size())

    # Пункт 2.2: підготовка графа та статична квантизація
    reader = ImageCalibrationReader(calibration_rows, data_dir, preprocess)
    quant_pre_process(
        input_model="lab4_fp32.onnx",
        output_model_path="lab4_fp32_preprocessed.onnx",
        skip_symbolic_shape=True,
    )
    onnx.checker.check_model("lab4_fp32_preprocessed.onnx")
    start_quant = time.perf_counter()
    quantize_static(
        model_input="lab4_fp32_preprocessed.onnx",
        model_output="lab4_int8.onnx",
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,       # із таблиці
        activation_type=QuantType.QUInt8,   # із таблиці
        weight_type=QuantType.QInt8,        # із таблиці
        per_channel=True,
        calibrate_method=CalibrationMethod.MinMax,
    )
    quant_seconds = time.perf_counter() - start_quant
    print(f"Час квантизації: {quant_seconds:.3f} с")
    onnx.checker.check_model("lab4_int8.onnx")
    # Пункт 3.1: логіти INT8 на всьому test
    session_int8 = ort.InferenceSession(
        "lab4_int8.onnx",
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )

    int8_batches = []
    for start in range(0, len(test_rows), 64):
        batch_rows = test_rows.iloc[start:start + 64]
        x, _ = load_test_batch(batch_rows, data_dir, preprocess, class_to_idx)
        batch_logits = session_int8.run(
            ["logits"], {"input": x.numpy()}
        )[0]
        int8_batches.append(batch_logits)

    logits_int8 = np.concatenate(int8_batches, axis=0)
    print("Форма логітів INT8:", logits_int8.shape)
    pred_int8 = logits_int8.argmax(axis=1)
    agreement_int8_fp32 = np.mean(pred_int8 == max_logit_onnx)
    print(f"Частка збігів INT8 та FP32: {agreement_int8_fp32:.6f}")
    f1_int8 = f1_score(
        y_true, pred_int8,
        labels=range(6), average="macro", zero_division=0,
    )
    print(f"F1 INT8: {f1_int8:.6f}")
    accuracy_int8 = np.mean(pred_int8 == y_true)
    print(f"Точність INT8: {accuracy_int8:.6f}")
    accuracy_fp32 = np.mean(max_logit_onnx == y_true)
    print(f"Точність FP32: {accuracy_fp32:.6f}")
    fp32_correct = max_logit_onnx == y_true
    int8_correct = pred_int8 == y_true
    b = np.count_nonzero(int8_correct & ~fp32_correct)
    c = np.count_nonzero(~int8_correct & fp32_correct)
    print(f"INT8 вірно, FP32 невірно: {b}")
    print(f"INT8 невірно, FP32 вірно: {c}")
    # Пункт 3.2: парна різниця accuracy та її похибка
    n = len(y_true)
    delta_acc = (b - c) / n
    se_delta = np.sqrt(b + c) / n
    print(f"Δ accuracy: {delta_acc:.6f} ± {se_delta:.6f}")
    noise_limit = 2 * np.sqrt(b + c)
    se_single = np.sqrt(accuracy_fp32 * (1 - accuracy_fp32) / n)
    if abs(b - c) <= noise_limit:
        print("Різниця в межах шуму, статистично незначуща")
    else:
        print("Різниця статистично значуща")
    print(f"noise_limit: {noise_limit:.6f}")
    print(f"Похибка однієї моделі: {se_single:.6f}")

    # Пункт 4.1: три найбільші r, медіана за згортками та втрата бітів
    conv_ratios = conv_weight_range_ratios(graph)
    print(f"Кількість згорток: {len(conv_ratios)}")
    for name, r in conv_ratios[:3]:
        print(f"{name}: r = {r:.4f}")
    median_r = float(np.median([r for _, r in conv_ratios]))
    print(f"Медіана r: {median_r:.4f}")
    print(f"log2(медіани r): {np.log2(median_r):.2f} біта")
    # Пункт 4.2: INT8 із потензорними вагами
    reader_per_tensor = ImageCalibrationReader(calibration_rows, data_dir, preprocess)

    quantize_static(
        model_input="lab4_fp32_preprocessed.onnx",
        model_output="lab4_int8_per_tensor.onnx",
        calibration_data_reader=reader_per_tensor,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=False,  # єдина змінена настройка
        calibrate_method=CalibrationMethod.MinMax,
    )
    onnx.checker.check_model("lab4_int8_per_tensor.onnx")

    # Пункт 4.3: якість потензорної моделі на тому самому test
    session_per_tensor = ort.InferenceSession(
        "lab4_int8_per_tensor.onnx",
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )

    per_tensor_batches = []
    for start in range(0, len(test_rows), 64):
        batch_rows = test_rows.iloc[start:start + 64]
        x, _ = load_test_batch(batch_rows, data_dir, preprocess, class_to_idx)
        batch_logits = session_per_tensor.run(
            ["logits"], {"input": x.numpy()}
        )[0]
        per_tensor_batches.append(batch_logits)

    logits_per_tensor = np.concatenate(per_tensor_batches, axis=0)
    print("Форма логітів INT8 per-tensor:", logits_per_tensor.shape)
    pred_per_tensor = logits_per_tensor.argmax(axis=1)
    agreement_per_tensor = np.mean(pred_per_tensor == max_logit_onnx)
    print(f"Згода INT8 per-tensor з FP32: {agreement_per_tensor:.6f}")
    f1_per_tensor = f1_score(
        y_true, pred_per_tensor,
        labels=range(6), average="macro", zero_division=0,
    )
    print(f"F1 INT8 per-tensor: {f1_per_tensor:.6f}")
    accuracy_per_tensor = np.mean(pred_per_tensor == y_true)
    print(f"Точність INT8 per-tensor: {accuracy_per_tensor:.6f}")
    per_tensor_correct = pred_per_tensor == y_true
    b_per_tensor = np.count_nonzero(per_tensor_correct & ~fp32_correct)
    c_per_tensor = np.count_nonzero(~per_tensor_correct & fp32_correct)
    print(f"INT8 per-tensor вірно, FP32 невірно: {b_per_tensor}")
    print(f"INT8 per-tensor невірно, FP32 вірно: {c_per_tensor}")
    delta_per_tensor = (b_per_tensor - c_per_tensor) / n
    se_per_tensor = np.sqrt(b_per_tensor + c_per_tensor) / n
    print(f"Δ accuracy INT8 per-tensor: {delta_per_tensor:.6f} ± {se_per_tensor:.6f}")
    # Пункт 5.1: один спільний вхід для вимірювання затримки
    x_one, _ = load_test_batch(
        test_rows.iloc[:1], data_dir, preprocess, class_to_idx
    )
    x_latency = x_one.numpy()
    print("Вхід для затримки:", x_latency.shape, x_latency.dtype)

    # Пункт 5.2: три раунди, по 20 прогрівів і 100 вимірів на варіант
    latency = benchmark_sessions(session, session_int8, x_latency)
    for variant, model_path in (
        ("FP32", Path("lab4_fp32.onnx")),
        ("INT8", Path("lab4_int8.onnx")),
    ):
        result = latency[variant]
        size_mb = model_path.stat().st_size / 1_000_000
        round_medians = ", ".join(f"{value:.3f}" for value in result["round_p50_ms"])
        print(
            f"{variant}: файл {size_mb:.2f} МБ, p50 {result['p50_ms']:.3f} мс, "
            f"p95 {result['p95_ms']:.3f} мс, p50 за раундами [{round_medians}] мс"
        )

    speedup = latency["FP32"]["p50_ms"] / latency["INT8"]["p50_ms"]
    print(f"Прискорення за p50: {speedup:.3f}×")
    fp32_rounds = latency["FP32"]["round_p50_ms"]
    int8_rounds = latency["INT8"]["round_p50_ms"]
    intervals_disjoint = (
        max(fp32_rounds) < min(int8_rounds)
        or max(int8_rounds) < min(fp32_rounds)
    )
    print(f"Діапазони p50 за раундами не перетинаються: {intervals_disjoint}")
    environment = runtime_environment(options)
    print("Середовище вимірювання:", environment)

    # Підсумок для відтворення й здачі: усі числа пунктів 1–5
    results = {
        "inputs": {
            "model": str(args.model),
            "split_csv": str(args.split),
            "data_dir": str(args.data_dir),
        },
        "environment": environment,
        "split_control": json.loads(summary.to_json(orient="records")),
        "point_1_fp32": {
            "export_seconds": float(export_seconds),
            "max_abs_logit_diff": float(diff),
            "prediction_agreement": float(agreement),
            "macro_f1_pytorch": float(f1_pytorch),
            "macro_f1_onnx": float(f1_onnx),
            "conv_nodes": node_types.count("Conv"),
            "batch_normalization_nodes": node_types.count("BatchNormalization"),
        },
        "point_2_quantization": {
            "calibration_count": len(calibration_rows),
            "calibration_file_sum": int(
                calibration_rows["file"].map(lambda name: int(Path(name).stem)).sum()
            ),
            "calibration_class_counts": {
                label: int(count)
                for label, count in calibration_rows.groupby("label").size().items()
            },
            "quantization_seconds": float(quant_seconds),
            "quant_format": "QDQ",
            "activation_type": "QUInt8",
            "weight_type": "QInt8",
            "calibrate_method": "MinMax",
            "per_channel": True,
        },
        "point_3_quality": {
            "fp32": {
                "macro_f1": float(f1_onnx),
                "accuracy": float(accuracy_fp32),
            },
            "int8_per_channel": {
                "macro_f1": float(f1_int8),
                "accuracy": float(accuracy_int8),
                "agreement_with_fp32": float(agreement_int8_fp32),
                "b": int(b),
                "c": int(c),
                "delta_accuracy": float(delta_acc),
                "se_delta_accuracy": float(se_delta),
                "within_noise_by_paired_rule": bool(abs(b - c) <= noise_limit),
            },
            "single_model_accuracy_se": float(se_single),
        },
        "point_4_per_tensor": {
            "top_three_conv_ratios": [
                {"node": name, "r": ratio} for name, ratio in conv_ratios[:3]
            ],
            "median_conv_ratio": median_r,
            "median_channel_bit_cost": float(np.log2(median_r)),
            "macro_f1": float(f1_per_tensor),
            "accuracy": float(accuracy_per_tensor),
            "agreement_with_fp32": float(agreement_per_tensor),
            "b": int(b_per_tensor),
            "c": int(c_per_tensor),
            "delta_accuracy": float(delta_per_tensor),
            "se_delta_accuracy": float(se_per_tensor),
            "within_noise_by_paired_rule": bool(
                abs(b_per_tensor - c_per_tensor)
                <= 2 * np.sqrt(b_per_tensor + c_per_tensor)
            ),
        },
        "point_5_latency": {
            "protocol": {
                "batch_size": 1,
                "rounds": 3,
                "warmup_calls_per_round": 20,
                "measured_calls_per_round": 100,
                "unit": "ms",
                "round_order": ["FP32", "INT8"],
            },
            "fp32": {
                **latency["FP32"],
                "file_bytes": Path("lab4_fp32.onnx").stat().st_size,
            },
            "int8": {
                **latency["INT8"],
                "file_bytes": Path("lab4_int8.onnx").stat().st_size,
            },
            "speedup_by_p50": float(speedup),
            "round_p50_intervals_disjoint": bool(intervals_disjoint),
        },
    }
    write_results_json(args.results, results)
    print(f"Результати записано: {args.results}")

if __name__ == "__main__":
    main(parse_args())
