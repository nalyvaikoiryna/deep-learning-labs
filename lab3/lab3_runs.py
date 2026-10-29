"""

Використовує функції з lab3.py, виправлені за lab3_fixes.patch.
Приклад: uv run lab3_runs.py check --data-dir .
Перевірки, сім запусків і вибір моделі за validation.
Оцінка test виконується окремою командою lab3_analysis.py.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lab3 import (
    classes, train_transform, validation_transform, SplitDataset, split,
    resolve_image_dir,
    build_scratch, build_frozen, build_unfreeze, train_one_epoch, validate,
    check_split, save_batch_preview, check_initial_loss,
    check_parameters, check_changes, check_bn_mistake,
)


import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from enum import Enum
import torchvision

SCRIPT_PATH = Path(__file__).resolve()
from lab3_results import RUN_PLAN, load_run_logs, save_selected_model
BUILDERS = {"scratch": build_scratch, "frozen": build_frozen, "unfreeze": build_unfreeze}


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def json_default(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Не вдалося записати в JSON тип {type(value).__name__}")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=json_default),
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_metadata():
    def git(*arguments):
        return subprocess.check_output(
            ["git", *arguments], cwd=SCRIPT_PATH.parent,
            text=True, stderr=subprocess.PIPE,
        ).strip()
    try:
        root = Path(git("rev-parse", "--show-toplevel")).resolve()
        commit = git("rev-parse", "HEAD")
        dirty = bool(git("status", "--porcelain", "--untracked-files=normal"))
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "Для основних запусків потрібен Git-репозиторій із хоча б одним комітом. "
            "Збережи цей скрипт у своїй теці lab3 в репозиторії."
        ) from exc
    if not SCRIPT_PATH.is_relative_to(root):
        raise RuntimeError("Скрипт має бути всередині репозиторію, коміт якого записується в журнал")
    return {"commit": commit, "dirty": dirty}


def device_name(device):
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    chip = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        try:
            chip = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return f"{chip} ({device.type})"


def transform_settings(pipeline):
    # Записуємо й неназвані в умові стандартні параметри: ratio, p,
    # interpolation, antialias, max_size, inplace тощо.
    return [
        {
            "type": type(transform).__name__,
            "repr": repr(transform),
            "parameters": {
                name: value for name, value in vars(transform).items()
                if not name.startswith("_") and name != "training"
            },
        }
        for transform in pipeline.transforms
    ]


def loader_settings(loader):
    fields = (
        "batch_size", "drop_last", "num_workers", "pin_memory", "timeout",
        "persistent_workers", "prefetch_factor", "in_order",
    )
    settings = {name: getattr(loader, name) for name in fields if hasattr(loader, name)}
    settings.update({
        "shuffle": isinstance(loader.sampler, torch.utils.data.RandomSampler),
        "generator": None,
        "sampler": type(loader.sampler).__name__,
        "custom_sampler": False,
        "worker_init_fn": None,
        "collate_fn": "torch.utils.data.default_collate",
        "multiprocessing_context": None,
    })
    return settings


def run_settings(model, optimizer, regime, train_loader, val_loader):
    parameter_names = {id(parameter): name for name, parameter in model.named_parameters()}
    groups = []
    for group in optimizer.param_groups:
        entry = {key: value for key, value in group.items() if key != "params"}
        entry["parameter_names"] = [parameter_names[id(p)] for p in group["params"]]
        entry["num_parameters"] = sum(p.numel() for p in group["params"])
        groups.append(entry)
    frozen = [] if regime == "scratch" else ["conv1", "bn1", "layer1", "layer2", "layer3"]
    if regime == "frozen":
        frozen.append("layer4")
    criterion = nn.CrossEntropyLoss()
    return {
        "model": "resnet18",
        "weights": None if regime == "scratch" else "ResNet18_Weights.IMAGENET1K_V1",
        "fc": {"in_features": 512, "out_features": 6, "bias": True, "initialization": "PyTorch default"},
        "initialization_order": ["torch.manual_seed", "np.random.seed", "resnet18", "replace_fc", "requires_grad", "to_device", "AdamW"],
        "classes": classes,
        "img_size": 160,
        "image_mode": "RGB",
        "dtype": "float32",
        "trainable_parameters": [name for name, p in model.named_parameters() if p.requires_grad],
        "eval_modules_during_training": frozen,
        "batch_norm": {
            name: {"eps": module.eps, "momentum": module.momentum,
                   "affine": module.affine, "track_running_stats": module.track_running_stats}
            for name, module in model.named_modules() if isinstance(module, nn.BatchNorm2d)
        },
        "optimizer": {"type": "AdamW", "defaults": optimizer.defaults, "groups": groups},
        "transforms": {"train": transform_settings(train_transform), "validation": transform_settings(validation_transform)},
        "loaders": {"train": loader_settings(train_loader), "validation": loader_settings(val_loader)},
        "loss": {"type": "CrossEntropyLoss", "weight": None, "ignore_index": criterion.ignore_index,
                 "reduction": criterion.reduction, "label_smoothing": criterion.label_smoothing},
        "validation_metric": {"type": "sklearn.metrics.f1_score", "average": "macro",
                              "labels": None, "sample_weight": None, "zero_division": "warn"},
        "epochs": 8,
        "steps_per_epoch": len(train_loader),
        "optimizer_steps": 8 * len(train_loader),
        "last_train_batch_size": 16,
        "train_loss_aggregation": "sum(batch_loss * batch_size) / 1200",
        "scheduler": None, "gradient_clipping": None, "mixed_precision": False,
        "early_stopping": False, "checkpoint_epoch": 8,
        "test_used": False,
        "timing_scope": "Sum of 8 training+validation epochs; excludes model setup, logging, and saving",
    }


def make_loaders(data_dir, csv_path):
    train_dir = resolve_image_dir(data_dir, "seg_train")
    test_dir = resolve_image_dir(data_dir, "seg_test")
    train_dataset = SplitDataset(csv_path, "train", train_dir, classes, train_transform)
    val_dataset = SplitDataset(csv_path, "validation", test_dir, classes, validation_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    if (len(train_dataset), len(val_dataset), len(train_loader)) != (1200, 1498, 38):
        raise ValueError("Очікується 1200 train, 1498 validation та 38 пакетів train")
    return train_loader, val_loader


def make_model(regime, seed ):
    return BUILDERS[regime]( seed, classes)


def run_checks(data_dir, output_dir, train_loader):
    checks_dir = output_dir / "checks"
    checks_dir.mkdir(parents=True, exist_ok=True)
    result = {}

    control = check_split(output_dir / "split.csv")
    control.to_csv(checks_dir / "split_control.csv", index=False)
    result["split"] = control.to_dict(orient="records")
    save_batch_preview(train_loader, classes, checks_dir / "batch_preview.png")

    result["initial_loss"] = []
    for regime in ("scratch", "frozen"):
        model, optimizer, device = make_model(regime, 1)
        loss = check_initial_loss(model, train_loader, device, regime)
        row = {"regime": regime, "seed": 1, "loss": loss,
               "ln6": float(np.log(6)), "difference": loss - float(np.log(6))}
        result["initial_loss"].append(row)
        print(f"Початкова втрата {regime}: {loss:.6f}; ln(6)={np.log(6):.6f}")
        del model, optimizer

    result["parameters"] = []
    for regime in ("scratch", "frozen", "unfreeze"):
        model, optimizer, device = make_model(regime, 1)
        report = check_parameters(model, optimizer, regime)
        result["parameters"].append(report)
        print(report)
        del model, optimizer

    result["changes"] = []
    for regime in ("frozen", "unfreeze"):
        model, optimizer, device = make_model(regime, 1)
        report = check_changes(model, optimizer, train_loader, device, regime)
        result["changes"].extend(report)
        print(pd.DataFrame(report).to_string(index=False))
        del model, optimizer

    # Прогноз зберігаємо ПЕРЕД дослідом. За наступних запусків читаємо його з файлу.
    prediction_path = checks_dir / "bn_prediction.txt"
    prediction = prediction_path.read_text(encoding="utf-8").strip() if prediction_path.exists() else ""
    if not prediction:
        prediction = input(
            "Перед дослідом BN запиши свій прогноз: що зміниться в параметрах та BN "
            "замороженого кістяка і чи буде це видно з train loss?\n> "
        ).strip()
        if not prediction:
            raise ValueError("Потрібен власний прогноз перед дослідом BN")
        prediction_path.write_text(prediction + "\n", encoding="utf-8")

    model, optimizer, device = make_model("frozen", 1)
    mistake = check_bn_mistake(model, optimizer, train_loader, device, "frozen")
    del model, optimizer
    result["bn_prediction"] = prediction
    result["bn_mistake"] = mistake
    print(pd.DataFrame(mistake["modules"]).to_string(index=False))
    print("Максимальна зміна running_mean:", mistake["running_mean_max_change"])
    print("Максимальна зміна running_var:", mistake["running_var_max_change"])
    print("Train loss перших п'яти пакетів:", mistake["train_loss_first_5_batches"])
    write_json(checks_dir / "results.json", result)
    # В очікуваному досліді параметри кістяка лишаються незмінними,
    # а BN-буфери оновлюються. Це перевірка поведінки коду, не оцінка прогнозу студента.
    if not all(
        row["parameters_changed"] == (row["module"] == "fc")
        and (row["bn_changed"] is None or row["bn_changed"])
        for row in mistake["modules"]
    ):
        raise ValueError("Дослід із BN дав неочікувану поведінку; дивись checks/results.json")
    print("Перевірки завершено. Зображення пакета й пояснення вимірювань переглянь для звіту.")
    return result


def run_experiment(regime, seed, data_dir, output_dir, train_loader, val_loader):
    run_dir = output_dir / "runs" / f"{regime}_seed{seed}"
    if run_dir.exists():
        raise FileExistsError(f"{run_dir} вже існує. Для нової серії обери інший --output-dir")
    model, optimizer, device = make_model(regime, seed)
    check_parameters(model, optimizer, regime)
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("Навчання має використовувати float32")
    split_hash = sha256(output_dir / "split.csv")
    log = {
        "regime": regime, "seed": seed, "status": "running",
        "git": git_metadata(), "source_sha256": sha256(SCRIPT_PATH),
        "model_code_sha256": sha256(SCRIPT_PATH.with_name("lab3.py")),
        "split_sha256": split_hash,
        "runtime": {"python": platform.python_version(), "torch": str(torch.__version__),
                    "torchvision": str(torchvision.__version__), "numpy": str(np.__version__),
                    "device": str(device), "device_name": device_name(device)},
        "config": run_settings(model, optimizer, regime, train_loader, val_loader),
        "history": [], "training_seconds": 0.0,
    }
    write_json(run_dir / "run.json", log)
    try:
        for epoch in range(1, 9):
            synchronize(device)
            started = time.perf_counter()
            train_loss = train_one_epoch(model, train_loader, optimizer, device, regime)
            val_f1, val_logits = validate(model, val_loader, device)
            synchronize(device)
            log["training_seconds"] += time.perf_counter() - started
            log["history"].append({"epoch": epoch, "train_loss": float(train_loss), "val_macro_f1": float(val_f1)})
            write_json(run_dir / "run.json", log)
            print(f"{regime} seed={seed} | епоха {epoch}/8 | loss={train_loss:.5f} | val F1={val_f1:.5f}", flush=True)

        if sha256(output_dir / "split.csv") != split_hash:
            raise ValueError("split.csv змінився під час навчання")
        val_logits = np.asarray(val_logits, dtype=np.float32)
        if val_logits.shape != (1498, 6):
            raise ValueError(f"Неправильна форма логітів validation: {val_logits.shape}")
        np.save(run_dir / "val_logits.npy", val_logits)
        torch.save({
            "state_dict": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
            "regime": regime, "seed": seed, "classes": classes, "img_size": 160,
        }, run_dir / "checkpoint.pt")
        log["checkpoint_sha256"] = sha256(run_dir / "checkpoint.pt")
        log["status"] = "complete"
        write_json(run_dir / "run.json", log)
    except Exception as exc:
        log["status"] = "failed"
        log["error"] = str(exc)
        write_json(run_dir / "run.json", log)
        raise
    return log


def summarize_runs(output_dir):
    """Лише читає сім журналів. Навчання та оцінки test тут немає."""
    logs = load_run_logs(output_dir)
    records = []
    for number, log in enumerate(logs, 1):
        last = log["history"][-1]
        records.append({"number": number, "regime": log["regime"], "seed": log["seed"],
                        "train_loss_epoch8": last["train_loss"],
                        "val_macro_f1_epoch8": last["val_macro_f1"],
                        "training_seconds": log["training_seconds"]})

    summary = pd.DataFrame(records)
    summary.to_csv(output_dir / "runs_summary.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"scratch": "#4C78A8", "frozen": "#F58518", "unfreeze": "#54A24B"}
    styles = {1: "-", 2: "--", 3: ":"}
    for log in logs:
        history = log["history"]
        epochs = [row["epoch"] for row in history]
        kwargs = {"label": f"{log['regime']} seed {log['seed']}",
                  "color": colors[log["regime"]], "linestyle": styles[log["seed"]]}
        axes[0].plot(epochs, [row["train_loss"] for row in history], **kwargs)
        axes[1].plot(epochs, [row["val_macro_f1"] for row in history], **kwargs)
    for ax, title, ylabel in zip(axes, ["Навчальна втрата", "Якість на validation"], ["Cross-entropy", "macro-F1"]):
        ax.set(title=title, xlabel="Епоха", ylabel=ylabel)
        ax.set_xticks(range(1, 9))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=160)
    plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser(description="ЛР3: перевірки та сім навчальних запусків")
    parser.add_argument("command", choices=["check", "train", "summary", "finalize"])
    parser.add_argument("--data-dir", type=Path, help="Тека, що містить seg_train і seg_test")
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "finalize":
        save_selected_model(output_dir)
        return
    if args.command == "summary":
        summarize_runs(output_dir)
        return
    if args.data_dir is None:
        parser.error("Для check/train потрібен --data-dir")
    data_dir = args.data_dir.expanduser().resolve()
    train_dir = resolve_image_dir(data_dir, "seg_train")
    test_dir = resolve_image_dir(data_dir, "seg_test")
    if args.command == "train":
        git_metadata()
        existing = [output_dir / "runs" / f"{r}_seed{s}" for r, s in RUN_PLAN]
        if any(path.exists() for path in existing):
            raise FileExistsError("У цій теці вже є запуски. Для повторної таблиці використовуй summary; для нової серії — інший --output-dir")
    csv_path = output_dir / "split.csv"
    if not csv_path.exists():
        split(train_dir, test_dir, classes, csv_path=csv_path)
    train_loader, val_loader = make_loaders(data_dir, csv_path)
    run_checks(data_dir, output_dir, train_loader)
    if args.command == "check":
        return
    for regime, seed in RUN_PLAN:
        run_experiment(regime, seed, data_dir, output_dir, train_loader, val_loader)
    summarize_runs(output_dir)
    save_selected_model(output_dir)


if __name__ == "__main__":
    main()
