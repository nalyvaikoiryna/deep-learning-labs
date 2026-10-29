from pathlib import Path
import torch
import json
import numpy as np
from sklearn import metrics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import platform
from lab3_runs import SplitDataset, classes, validation_transform, summarize_runs, write_json
from lab3_results import (
    load_runs, calculate_statistics, select_best_model, load_selected_model, file_sha256,
)
from lab3 import resolve_image_dir
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision.models import resnet18
import argparse

def compare_regimes(results, statistics):
    frozen = None
    unfreeze = None
    scratch = None

    for row in statistics:
        if row["regime"] == "frozen":
            frozen = row
        elif row["regime"] == "unfreeze":
            unfreeze = row
        elif row["regime"] == "scratch":
            scratch = row

    if scratch is None or frozen is None or unfreeze is None:
        raise ValueError("Не знайдено статистику всіх трьох режимів")

    # 1. Різниця середніх F1
    delta_pre = frozen["mean_f1"] - scratch["mean_f1"]
    delta_unf = unfreeze["mean_f1"] - frozen["mean_f1"]

    delta_pre_pp = delta_pre * 100
    delta_unf_pp = delta_unf * 100
    # 2. Найбільший розмах між seed
    max_range = max(frozen["range_f1"], unfreeze["range_f1"])
    
    # 3. Визначити, який режим кращий
    

    winner = max(
        [scratch, frozen, unfreeze],
        key=lambda row: row["mean_f1"]
    )["regime"]

    # 4. Чи більший ефект за розкид?
    effect_larger = True if abs(delta_unf) > max_range else False

    # 5. Вивести результати
    print("\nСередні значення macro-F1:")
    print(f"Scratch:  {scratch['mean_f1']:.4f}")
    print(f"Frozen:   {frozen['mean_f1']:.4f}")
    print(f"Unfreeze: {unfreeze['mean_f1']:.4f}")

    print("\nРозкид між seed:")
    print(f"Frozen:   {frozen['range_f1']:.4f}")
    print(f"Unfreeze: {unfreeze['range_f1']:.4f}")
    print(f"R = {max_range:.4f}")

    print("\nЕфекти:")
    print(f"Δpre = {delta_pre:+.4f} ({delta_pre_pp:+.2f} в.п.)")
    print(f"Δunf = {delta_unf:+.4f} ({delta_unf_pp:+.2f} в.п.)")

    print(f"\nНайкращий режим: {winner}")
    print(f"Ефект розморожування більший за розкид: {effect_larger}")

    return {
    "delta_pre": delta_pre,
    "delta_unf": delta_unf,
    "delta_pre_pp": delta_pre_pp,
    "delta_unf_pp": delta_unf_pp,
    "winner": winner,
    "max_range": max_range,
    "effect_larger": effect_larger,
    "pre_effect_larger": abs(delta_pre) > max_range,
    "total_gain": delta_pre + delta_unf,
    }
def evaluate_test(model, test_loader, device):
    if device.type != "cpu":
        raise ValueError("За протоколом test обчислюється тільки на CPU")
    model = model.to(device)
    model.eval()
    all_logits, all_labels = [], []
    with torch.inference_mode():
        for batch, (images, labels) in enumerate(test_loader, 1):
            logits = model(images.to(device))
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            if batch % 4 == 0 or batch == len(test_loader):
                print(f"Test CPU: пакет {batch}/{len(test_loader)}", flush=True)
    if not all_logits:
        raise ValueError("Порожня test-вибірка")
    logits = torch.cat(all_logits).numpy()
    y_true = torch.cat(all_labels).numpy()
    if logits.shape != (len(y_true), len(classes)) or not np.isfinite(logits).all():
        raise ValueError("Некоректні логіти test")
    y_pred = logits.argmax(axis=1)
    macro_f1 = metrics.f1_score(y_true, y_pred, labels=range(6), average="macro", zero_division=0)
    return float(macro_f1), y_true, y_pred, logits


def build_confusion_matrix(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("Кількість міток не збігається")
    return metrics.confusion_matrix(y_true, y_pred, labels=range(6)).tolist()


def find_five_errors(matrix, y_true, y_pred, filenames):
    if not len(y_true) == len(y_pred) == len(filenames):
        raise ValueError("Кількості міток і файлів не збігаються")
    # За рівної частоти: менший індекс справжнього, потім прогнозованого класу.
    pairs = sorted(
        [(true, pred) for true in range(6) for pred in range(6)
         if true != pred and matrix[true][pred] > 0],
        key=lambda pair: (-matrix[pair[0]][pair[1]], pair[0], pair[1]),
    )
    indices = {
        pair: sorted(
            [i for i, (true, pred) in enumerate(zip(y_true, y_pred)) if (true, pred) == pair],
            key=lambda i: str(filenames[i]),
        )
        for pair in pairs
    }
    selected = []
    rank = 0
    while len(selected) < 5:
        added = False
        for true, pred in pairs:
            group = indices[true, pred]
            if rank < len(group):
                selected.append({"index": group[rank], "true": true, "pred": pred,
                                 "frequency": int(matrix[true][pred])})
                added = True
                if len(selected) == 5:
                    break
        if not added:
            break
        rank += 1
    return selected


def save_confusion_plot(matrix, output_path):
    values = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(8, 7))
    display = ax.imshow(values, cmap="Blues")
    fig.colorbar(display, ax=ax)
    ax.set(xticks=range(6), yticks=range(6), xticklabels=classes, yticklabels=classes,
           xlabel="Передбачений клас", ylabel="Справжній клас", title="Матриця плутанини: test")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    for true in range(6):
        for pred in range(6):
            ax.text(pred, true, str(values[true, pred]), ha="center", va="center",
                    color="white" if values[true, pred] > values.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_error_plot(errors, test_dataset, output_path):
    from PIL import Image
    fig, axes = plt.subplots(1, 5, figsize=(16, 4.8), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, error in zip(axes.flat, errors):
        path = test_dataset.image_dir / error["true_class"] / error["file"]
        with Image.open(path) as image:
            ax.imshow(image.convert("RGB"))
        ax.set_title(f"{error['file']}\n{error['true_class']} → {error['pred_class']}\n"
                     f"p={error['probability']:.3f}; кількість={error['frequency']}", fontsize=10)
    if not errors:
        axes[0, 2].text(0.5, 0.5, "Помилок немає", ha="center", va="center")
    fig.tight_layout(pad=2.0)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines += ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join(lines)


def save_analysis_report(output_dir, results, statistics, comparison, selected,
                         test_dataset, macro_f1, y_true, y_pred, logits, matrix, errors):
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    for error in errors:
        row = test_dataset.rows.iloc[error["index"]]
        error.update({"file": str(row["file"]), "true_class": classes[error["true"]],
                      "pred_class": classes[error["pred"]],
                      "probability": float(probabilities[error["index"], error["pred"]])})
    accuracy = float(metrics.accuracy_score(y_true, y_pred))
    test_results = {
        "macro_f1": macro_f1, "accuracy": accuracy, "num_images": len(y_true),
        "accuracy_standard_error": float(np.sqrt(accuracy * (1 - accuracy) / len(y_true))),
        "test_minus_validation_f1": macro_f1 - selected["val_macro_f1"],
        "device": "cpu",
    }
    report = {
        "split_sha256": file_sha256(output_dir / "split.csv"),
        "model_sha256": file_sha256(output_dir / "lab3_model.pt"),
        "runtime": {"python": platform.python_version(), "torch": str(torch.__version__),
                    "device": "cpu"},
        "runs": results, "statistics": statistics, "effects": comparison,
        "selected": {key: selected[key] for key in ("regime", "seed", "val_macro_f1")},
        "test": test_results, "confusion_matrix": matrix, "errors": errors,
    }
    write_json(analysis_dir / "results.json", report)
    pd.DataFrame(statistics).to_csv(analysis_dir / "statistics.csv", index=False)
    matrix_table = pd.DataFrame(matrix, index=classes, columns=classes)
    matrix_table.to_csv(analysis_dir / "confusion_matrix.csv", index_label="true_class")
    pd.DataFrame(errors, columns=["file", "true_class", "pred_class", "probability", "frequency", "index", "true", "pred"]).to_csv(
        analysis_dir / "five_errors.csv", index=False)
    np.save(analysis_dir / "test_logits.npy", logits)
    pd.DataFrame({
        "file": test_dataset.rows["file"].to_list(),
        "true_class": [classes[i] for i in y_true],
        "pred_class": [classes[i] for i in y_pred],
        "probability": probabilities[np.arange(len(y_pred)), y_pred],
    }).to_csv(analysis_dir / "test_predictions.csv", index=False)
    save_confusion_plot(matrix, analysis_dir / "confusion_matrix.png")
    save_error_plot(errors, test_dataset, analysis_dir / "five_errors.png")

    effect_rows = [
        ["R", f"{comparison['max_range']:.6f}"],
        ["Δpre", f"{comparison['delta_pre']:+.6f}"],
        ["Δunf", f"{comparison['delta_unf']:+.6f}"],
        ["Δpre + Δunf", f"{comparison['total_gain']:+.6f}"],
    ]
    sections = [
        "# Результати команди аналізу",
        "## Сім запусків",
        markdown_table(["Режим", "Зерно", "Train loss, епоха 8", "Validation F1, епоха 8", "Час, с"],
                       [[r['regime'], r['seed'], f"{r['train_loss']:.6f}", f"{r['val_macro_f1']:.6f}", f"{r['training_seconds']:.3f}"] for r in results]),
        "![Криві навчання](../training_curves.png)",
        "## Середні значення та розмах",
        markdown_table(["Режим", "Середній F1", "Розмах"],
                       [[r['regime'], f"{r['mean_f1']:.6f}", 'не оцінено' if r['range_f1'] is None else f"{r['range_f1']:.6f}"] for r in statistics]),
        markdown_table(["Величина", "Значення"], effect_rows),
        f"Δpre: {'ефект вартий перевірки, але не доведений' if comparison['pre_effect_larger'] else 'різниця не відрізняється від розкиду'}.\n\n"
        f"Δunf: {'ефект вартий перевірки, але не доведений' if comparison['effect_larger'] else 'різниця не відрізняється від розкиду'}.",
        "## Вибір та test на CPU",
        f"Вибрано **{selected['regime']}, seed {selected['seed']}**, validation F1 = {selected['val_macro_f1']:.6f}.",
        markdown_table(["Метрика", "Значення"], [[key, f"{value:.6f}" if isinstance(value, float) else value] for key, value in test_results.items()]),
        "## Матриця плутанини",
        markdown_table(["Справжній / прогноз", *classes], [[name, *row] for name, row in zip(classes, matrix)]),
        "![Матриця плутанини](confusion_matrix.png)",
        "## Характерні помилки",
        markdown_table(["Файл", "Справжній клас", "Прогноз", "Імовірність", "Частота пари"],
                       [[r['file'], r['true_class'], r['pred_class'], f"{r['probability']:.6f}", r['frequency']] for r in errors]),
        "![П'ять помилок](five_errors.png)",
        "Для кожного зображення додайте власне пояснення: обидва класи у сцені, ймовірна помилка розмітки або помилка моделі на однозначному зображенні.",
    ]
    (analysis_dir / "report.md").write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    print(f"\nВибрано: {selected['regime']}, seed={selected['seed']}, validation F1={selected['val_macro_f1']:.6f}")
    print(f"Test CPU: macro-F1={macro_f1:.6f}, accuracy={accuracy:.6f}, SE accuracy={test_results['accuracy_standard_error']:.6f}")
    print(f"Test − validation F1: {test_results['test_minus_validation_f1']:+.6f}")
    print(matrix_table.to_string())
    print(pd.DataFrame(errors).to_string(index=False))
    print(f"\nТаблиці та рисунки: {analysis_dir}; текст звіту: {analysis_dir / 'report.md'}")
    return report


def main():
    parser = argparse.ArgumentParser(description="ЛР3: аналіз журналів і test на CPU")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()

    # Усі перевірки й вибір лише за validation виконуються до читання test.
    results = load_runs(output_dir)
    statistics = calculate_statistics(results)
    comparison = compare_regimes(results, statistics)
    selected = select_best_model(results, statistics, output_dir)
    checkpoint = load_selected_model(output_dir, selected)
    summarize_runs(output_dir)

    device = torch.device("cpu")
    model = resnet18(weights=None)
    model.fc = nn.Linear(512, len(classes))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model = model.to(device)

    test_dataset = SplitDataset(
        output_dir / "split.csv", "test", resolve_image_dir(data_dir, "seg_test"),
        classes, validation_transform,
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=False, num_workers=0)
    macro_f1, y_true, y_pred, logits = evaluate_test(model, test_loader, device)
    matrix = build_confusion_matrix(y_true, y_pred)
    errors = find_five_errors(matrix, y_true, y_pred, test_dataset.rows["file"].to_list())
    save_analysis_report(output_dir, results, statistics, comparison, selected,
                         test_dataset, macro_f1, y_true, y_pred, logits, matrix, errors)


if __name__ == "__main__":
    main()
