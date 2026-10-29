"""Спільні перевірки журналів і вибір моделі лише за validation."""
from pathlib import Path
import hashlib
import json
import shutil

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from lab3 import classes

RUN_PLAN = [
    ("scratch", 1),
    ("frozen", 1), ("frozen", 2), ("frozen", 3),
    ("unfreeze", 1), ("unfreeze", 2), ("unfreeze", 3),
]


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_run_logs(output_dir):
    output_dir = Path(output_dir)
    csv_path = output_dir / "split.csv"
    current_hash = file_sha256(csv_path)
    table = pd.read_csv(csv_path)
    if set(table.columns) != {"file", "label", "part"} or table.isna().any().any():
        raise ValueError("Неправильний формат split.csv")
    if table["file"].duplicated().any():
        raise ValueError("У split.csv є повтори файлів")
    counts = table.groupby("part").size().to_dict()
    if counts != {"train": 1200, "validation": 1498, "test": 1502}:
        raise ValueError("Неправильна кількість рядків частин split.csv")
    class_to_idx = {name: i for i, name in enumerate(classes)}
    if not table["label"].isin(classes).all():
        raise ValueError("Невідомий клас у split.csv")
    labels = table.loc[table["part"] == "validation", "label"].map(class_to_idx).to_numpy()

    logs = []
    for regime, seed in RUN_PLAN:
        run_dir = output_dir / "runs" / f"{regime}_seed{seed}"
        log = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        if log.get("status") != "complete" or (log.get("regime"), log.get("seed")) != (regime, seed):
            raise ValueError(f"Незавершений або невідповідний журнал: {run_dir}")
        if log.get("split_sha256") != current_hash:
            raise ValueError(f"Хеш split.csv не збігається: {run_dir}")
        if [row["epoch"] for row in log["history"]] != list(range(1, 9)):
            raise ValueError(f"Очікуються рівно епохи 1–8: {run_dir}")
        if log["config"]["classes"] != classes or log["config"]["img_size"] != 160:
            raise ValueError(f"Невідповідна конфігурація: {run_dir}")
        values = [log["training_seconds"]]
        for row in log["history"]:
            values.extend([row["train_loss"], row["val_macro_f1"]])
            if row["train_loss"] < 0 or not 0 <= row["val_macro_f1"] <= 1:
                raise ValueError(f"Неправильна метрика: {run_dir}")
        if not np.isfinite(values).all() or log["training_seconds"] < 0:
            raise ValueError(f"Некоректні числові результати: {run_dir}")
        logits = np.load(run_dir / "val_logits.npy", allow_pickle=False)
        if logits.shape != (1498, 6) or not np.isfinite(logits).all():
            raise ValueError(f"Неправильні логіти validation: {run_dir}")
        score = f1_score(labels, logits.argmax(axis=1), labels=range(6), average="macro", zero_division=0)
        if not np.isclose(score, log["history"][-1]["val_macro_f1"], atol=1e-12, rtol=0):
            raise ValueError(f"F1 журналу не відповідає логітам validation: {run_dir}")
        logs.append(log)
    return logs


def load_runs(output_dir):
    return [
        {
            "regime": log["regime"], "seed": log["seed"],
            "train_loss": log["history"][-1]["train_loss"],
            "val_macro_f1": log["history"][-1]["val_macro_f1"],
            "training_seconds": log["training_seconds"],
        }
        for log in load_run_logs(output_dir)
    ]


def calculate_statistics(results):
    statistics = []
    for regime in ["scratch", "frozen", "unfreeze"]:
        values = [run["val_macro_f1"] for run in results if run["regime"] == regime]
        statistics.append({
            "regime": regime,
            "mean_f1": sum(values) / len(values),
            "min_f1": min(values), "max_f1": max(values),
            "range_f1": max(values) - min(values) if len(values) > 1 else None,
        })
    return statistics


def select_best_model(results, statistics, output_dir):
    # При рівності max зберігає порядок: scratch/frozen/unfreeze, потім менше зерно.
    best_regime = max(statistics, key=lambda row: row["mean_f1"])["regime"]
    candidates = [run for run in results if run["regime"] == best_regime]
    best_run = max(candidates, key=lambda run: run["val_macro_f1"])
    return {
        "regime": best_regime, "seed": best_run["seed"],
        "val_macro_f1": best_run["val_macro_f1"],
        "checkpoint": str(Path(output_dir) / "runs" / f"{best_regime}_seed{best_run['seed']}" / "checkpoint.pt"),
    }


def check_checkpoint(checkpoint, selected):
    for key, value in {"regime": selected["regime"], "seed": selected["seed"],
                       "classes": classes, "img_size": 160}.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"Невідповідне поле {key} у збереженій моделі")
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict) or not state or not all(isinstance(t, torch.Tensor) for t in state.values()):
        raise ValueError("У моделі немає коректного state_dict")


def save_selected_model(output_dir):
    """Завершення train; також використовується finalize для старих журналів."""
    output_dir = Path(output_dir)
    results = load_runs(output_dir)
    selected = select_best_model(results, calculate_statistics(results), output_dir)
    source = Path(selected["checkpoint"])
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    check_checkpoint(checkpoint, selected)
    source_hash = file_sha256(source)
    source_log = json.loads(source.with_name("run.json").read_text(encoding="utf-8"))
    if source_log.get("checkpoint_sha256", source_hash) != source_hash:
        raise ValueError("Checkpoint змінився після навчання")
    target = output_dir / "lab3_model.pt"
    temporary = target.with_suffix(".pt.tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(target)
    selection = {key: selected[key] for key in ("regime", "seed", "val_macro_f1")}
    selection.update({"split_sha256": file_sha256(output_dir / "split.csv"),
                      "checkpoint_sha256": source_hash, "epoch": 8})
    temporary_json = output_dir / "selection.json.tmp"
    temporary_json.write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_json.replace(output_dir / "selection.json")
    print(f"Збережено {target}: {selected['regime']}, seed={selected['seed']}, val F1={selected['val_macro_f1']:.6f}")
    return selected


def load_selected_model(output_dir, selected):
    """Перевірка вибору до доступу до test; решта шести checkpoint не потрібні."""
    output_dir = Path(output_dir)
    path = output_dir / "lab3_model.pt"
    if not path.is_file() or not (output_dir / "selection.json").is_file():
        raise FileNotFoundError("Спочатку виконай lab3_runs.py finalize або повну команду train")
    selection = json.loads((output_dir / "selection.json").read_text(encoding="utf-8"))
    for key in ("regime", "seed", "val_macro_f1"):
        if selection.get(key) != selected[key]:
            raise ValueError("lab3_model.pt не відповідає вибору за validation")
    if selection.get("epoch") != 8 or selection.get("split_sha256") != file_sha256(output_dir / "split.csv"):
        raise ValueError("Невідповідний поділ або епоха у selection.json")
    digest = file_sha256(path)
    if digest != selection.get("checkpoint_sha256"):
        raise ValueError("lab3_model.pt змінився після вибору")
    original = Path(selected["checkpoint"])
    if original.is_file() and file_sha256(original) != digest:
        raise ValueError("lab3_model.pt відрізняється від вибраного checkpoint")
    log = json.loads(original.with_name("run.json").read_text(encoding="utf-8"))
    if log.get("checkpoint_sha256", digest) != digest:
        raise ValueError("Хеш моделі не збігається з журналом навчання")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    check_checkpoint(checkpoint, selected)
    return checkpoint
