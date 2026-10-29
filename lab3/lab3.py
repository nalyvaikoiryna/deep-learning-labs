from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import sklearn.metrics as metrics
import argparse
classes = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

def resolve_image_dir(data_dir, split_name):
    """Знайти папку класів у seg_train або seg_test, з одним чи двома рівнями."""
    if split_name not in ("seg_train", "seg_test"):
        raise ValueError(f"Невідома частина набору: {split_name}")

    root = Path(data_dir).expanduser().resolve()
    candidates = [root / split_name / split_name, root / split_name]
    if root.name == split_name:
        candidates.append(root)

    matches = []
    for candidate in candidates:
        if all((candidate / label).is_dir() for label in classes):
            resolved = candidate.resolve()
            if resolved not in matches:
                matches.append(resolved)

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        found = "\n".join(str(path) for path in matches)
        raise ValueError(f"Знайдено кілька папок {split_name}; уточни --data-dir:\n{found}")

    checked = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Не знайдено {split_name} з усіма шістьма папками класів. "
        f"Перевір --data-dir. Перевірені шляхи:\n{checked}"
    )


def split(train_dir, test_dir, classes, csv_path="split.csv"):
    parts = []
    rng = np.random.default_rng(0)
    
    for label in classes:
        class_dir = train_dir / label
        files = sorted(p.name for p in class_dir.glob("*.jpg"))

        idx = np.arange(len(files))
        rng.shuffle(idx)

        selected_files = [files[i] for i in idx[:200]]
        train_df = pd.DataFrame({"file": selected_files, "label": label, "part": "train"})
        parts.append(train_df)
    rng = np.random.default_rng(0)

    for label in classes:
        class_dir = test_dir / label
        files = sorted(p.name for p in class_dir.glob("*.jpg"))

        idx = np.arange(len(files))
        rng.shuffle(idx)

        cut = int(len(files) // 2)
        val_files = [files[i] for i in idx[:cut]]
        test_files = [files[i] for i in idx[cut:]]
        val_df = pd.DataFrame({"file": val_files, "label": label, "part": "validation"})
        test_df = pd.DataFrame({"file": test_files, "label": label, "part": "test"})
        parts.append(val_df)
        parts.append(test_df)
    
    full_df = pd.concat(parts, ignore_index=True)
    print("Кількість таблиць:", len(parts))
    print("Кількість зображень:", len(full_df))
    print(full_df.groupby("part").size())
    full_df.to_csv(csv_path, index=False)

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(160, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
])
validation_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
])
class SplitDataset(Dataset):
    def __init__(self, csv_path, part, image_dir, classes, transform):
        table = pd.read_csv(csv_path)

        self.rows = table[table["part"] == part].reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.class_to_idx = {
            name: i for i, name in enumerate(classes)
        }

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows.iloc[idx]

        image_path = self.image_dir / row["label"] / row["file"]
        label = self.class_to_idx[row["label"]]     # Твій рядок: числова мітка класу

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        image = self.transform(image)
        return image, label
    




def build_scratch(seed, classes):
    # Define the model
    W = None
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = resnet18(weights=W)

    model.fc = nn.Linear(512, len(classes))

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-3,
    weight_decay=1e-4
    )
    return model, optimizer, device
def build_unfreeze( seed, classes):
    # Define the model
    W = ResNet18_Weights.IMAGENET1K_V1
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = resnet18(weights=W)

    model.fc = nn.Linear(512, len(classes))
    for param in model.parameters():
        param.requires_grad = False

    # Дозволяємо навчатися лише останньому шару
    for param in model.fc.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(
    [
        {"params": model.fc.parameters(), "lr": 3e-3},
        {"params": model.layer4.parameters(), "lr": 3e-4},
    ],
    weight_decay=1e-4
    )
    return model, optimizer, device

def build_frozen(seed, classes):
    # Define the model
 
    W = ResNet18_Weights.IMAGENET1K_V1
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = resnet18(weights=W)

    model.fc = nn.Linear(512, len(classes))
    for param in model.parameters():
        param.requires_grad = False

    # Дозволяємо навчатися лише останньому шару
    for param in model.fc.parameters():
        param.requires_grad = True
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(
    model.fc.parameters(),
    lr=3e-3,
    weight_decay=1e-4
    )
    return model, optimizer, device

def set_train_mode(model, regime):
    model.train()

    if regime in ("frozen", "unfreeze"):
        frozen_modules = [
            model.conv1,
            model.bn1,
            model.layer1,
            model.layer2,
            model.layer3,
        ]

        for module in frozen_modules:
            module.eval()

        if regime == "frozen":
            model.layer4.eval()
def train_one_epoch(model, train_loader, optimizer, device, regime):
    set_train_mode(model, regime)
    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * images.size(0)

    return loss_sum / len(train_loader.dataset)
def validate(model, val_loader, device):
    model.eval()
    logits_batches = []
    labels_batches = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            logits_batches.append(logits.cpu())
            labels_batches.append(labels.cpu())

    all_logits = torch.cat(logits_batches, dim=0)
    all_labels = torch.cat(labels_batches, dim=0)

    predicted = all_logits.argmax(dim=1)

    macrof1 = metrics.f1_score(
        all_labels.numpy(),
        predicted.numpy(),
        average="macro"
    )

    return macrof1, all_logits.numpy()

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

import matplotlib.pyplot as plt
import torch

def save_batch_preview(train_loader, classes, output_path):
    images, labels = next(iter(train_loader))

    images = images[:16].cpu()
    labels = labels[:16].cpu()

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    images = (images * std + mean).clamp(0, 1)

    fig, axes = plt.subplots(4, 4, figsize=(10, 10))

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].permute(1, 2, 0))
        ax.set_title(classes[labels[i]])  # Назва  класу
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)



def check_initial_loss(model, train_loader, device, regime):
    set_train_mode(model, regime)
    images, labels = next(iter(train_loader))
    with torch.no_grad():
        logits = model(images.to(device))
        loss = nn.CrossEntropyLoss()(logits, labels.to(device))
    # Значення може відрізнятися від ln(6): це вимірювання, а не assert рівності.
    return float(loss.item())

def check_parameters(model, optimizer, regime):
    total = sum(p.numel() for p in model.parameters())

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    actual_groups = []

    for group in optimizer.param_groups:
        params = group["params"]

        assert all(p.requires_grad for p in params), (
            "В оптимізатор потрапили заморожені параметри"
        )

        count = sum(p.numel() for p in params)
        lr = group["lr"]
        actual_groups.append((count, lr))

    # Для unfreeze : спочатку fc, потім layer4
    expected_groups = {
        "scratch": [(11_179_590, 3e-3)],
        "frozen": [(3_078, 3e-3)],
        "unfreeze": [(3_078, 3e-3), (8_393_728, 3e-4)],
    }

    expected = expected_groups[regime]
    expected_trainable = sum(count for count, lr in expected)

    assert total == 11_179_590, (
        f"Загальна кількість параметрів неправильна: {total}"
    )
    assert trainable == expected_trainable, (
        f"Параметрів, що навчаються: {trainable}, "
        f"очікується {expected_trainable}"
    )
    assert actual_groups == expected, (
        f"Групи оптимізатора: {actual_groups}, очікується {expected}"
    )

    return {
        "regime": regime,
        "total": total,
        "trainable": trainable,
        "groups": actual_groups,
    }

from itertools import islice

def check_changes(model, optimizer, train_loader, device, regime):
    assert regime in ("frozen", "unfreeze")

    # Незалежна копія параметрів і буферів до навчання
    before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }

    set_train_mode(model, regime)
    criterion = nn.CrossEntropyLoss()

    # Рівно перші п'ять пакетів
    for images, labels in islice(train_loader, 5):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

    after = model.state_dict()
    parameter_names = set(dict(model.named_parameters()))

    modules = [
        "conv1", "bn1", "layer1",
        "layer2", "layer3", "layer4", "fc"
    ]
    bn_suffixes = ("running_mean", "running_var", "num_batches_tracked")
    report = []

    for module_name in modules:
        keys = [
            key for key in before
            if key.startswith(module_name + ".")
        ]

        param_keys = [key for key in keys if key in parameter_names]
        bn_keys = [key for key in keys if key.endswith(bn_suffixes)]

        params_changed = 1 if any(not torch.equal(before[key], after[key]) for key in param_keys) else 0
        bn_changed = 1 if any(not torch.equal(before[key], after[key]) for key in bn_keys) else 0 if bn_keys else None

        expected_change = (
            module_name == "fc"
            or (regime == "unfreeze" and module_name == "layer4")
        )

        matches_expected = (
            params_changed == expected_change
            and (bn_changed is None or bn_changed == expected_change)
        )

        report.append({
            "regime": regime,
            "module": module_name,
            "parameters_changed": params_changed,
            "bn_changed": bn_changed,
            "matches_expected": matches_expected,
        })

    assert all(row["matches_expected"] for row in report), report
    return report




def check_bn_mistake(model, optimizer, train_loader, device, regime):
    assert regime == "frozen"

    # Незалежна копія параметрів і буферів до навчання
    before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }

    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []

    # Рівно перші п'ять пакетів
    for images, labels in islice(train_loader, 5):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    after = model.state_dict()
    parameter_names = set(dict(model.named_parameters()))

    modules = [
        "conv1", "bn1", "layer1",
        "layer2", "layer3", "layer4", "fc"
    ]
    bn_suffixes = ("running_mean", "running_var", "num_batches_tracked")
    report = []

    for module_name in modules:
        keys = [
            key for key in before
            if key.startswith(module_name + ".")
        ]

        param_keys = [key for key in keys if key in parameter_names]
        bn_keys = [key for key in keys if key.endswith(bn_suffixes)]

        params_changed = 1 if any(not torch.equal(before[key], after[key]) for key in param_keys) else 0
        bn_changed = 1 if any(not torch.equal(before[key], after[key]) for key in bn_keys) else 0 if bn_keys else None

        mean_key = "layer1.0.bn1.running_mean"
        var_key = "layer1.0.bn1.running_var"

        mean_change = (after[mean_key] - before[mean_key]).abs().max().item()
        var_change = (after[var_key] - before[var_key]).abs().max().item()

        report.append({
        
            "regime": regime,
            "module": module_name,
            "parameters_changed": params_changed,
            "bn_changed": bn_changed,
            "mean_change": mean_change,
            "var_change": var_change,
        })

    return {
        "modules": report,
        "train_loss_first_5_batches": losses,
        "running_mean_max_change": float(mean_change),
        "running_var_max_change": float(var_change),
    }

if __name__ == "__main__":
    

    parser = argparse.ArgumentParser(description="ЛР3: підготувати поділ даних")
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    train_dir = resolve_image_dir(args.data_dir, "seg_train")
    test_dir = resolve_image_dir(args.data_dir, "seg_test")

    # Не основний main, але використовуався для перевірки поділу та завантажувачів
    if not Path("split.csv").exists():
        split(train_dir, test_dir, classes, csv_path="split.csv")

    # Завантажувачі тільки при прямому запуску цього файлу.

    train_dataset = SplitDataset(
        csv_path="split.csv",
        part="train",
        image_dir=train_dir,
        classes=classes,
        transform=train_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        drop_last=False,
        num_workers=0
    )

    val_dataset = SplitDataset(
        csv_path="split.csv",
        part="validation",
        image_dir=test_dir,
        classes=classes,
        transform=validation_transform
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )
    
    