"""
Quick Fine-tune ResNet18 on the local CIFAR-10 test folder.
Split: 80% train / 20% eval
Saves best checkpoint to: cifar10_finetuned.pth
Then runs Domain Shift evaluation and writes results to results_finetuned.txt
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms

from src.data.corruptions import apply_domain_shift, CORRUPTIONS_REGISTRY
from src.models.model import load_model, CIFARResNet18, ModelWrapper

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR   = os.path.join(os.path.dirname(__file__), "cifar10", "test")
CKPT_PATH  = os.path.join(os.path.dirname(__file__), "cifar10_finetuned.pth")
RESULT_FILE = os.path.join(os.path.dirname(__file__), "results_finetuned.txt")
EPOCHS     = 15
BATCH_SIZE = 128
LR         = 0.05

def get_loaders():
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    val_tf = transforms.Compose([transforms.ToTensor()])

    full_ds = torchvision.datasets.ImageFolder(DATA_DIR, transform=train_tf)
    n_train = int(0.8 * len(full_ds))
    n_val   = len(full_ds) - n_train
    train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    # Apply val transform to val split
    val_ds.dataset = torchvision.datasets.ImageFolder(DATA_DIR, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, val_loader

def train():
    print(f"Device: {DEVICE}")
    print("Loading data (80% train / 20% val) from cifar10/test ...")
    train_loader, val_loader = get_loaders()

    # Build fresh CIFAR ResNet-18 with He init (default PyTorch)
    backbone = CIFARResNet18(num_classes=10)
    model    = ModelWrapper(backbone, device=DEVICE)

    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    print(f"\nFine-tuning for {EPOCHS} epochs ...\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits, _ = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()

        # Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                logits, _ = model(imgs)
                preds = logits.argmax(1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
        val_acc = correct / total * 100
        print(f"Epoch [{epoch:>2}/{EPOCHS}] | Loss: {running_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.backbone.state_dict(), CKPT_PATH)
            print(f"  => Saved best model (acc={best_acc:.2f}%)")

    print(f"\nBest Val Accuracy: {best_acc:.2f}%")
    return best_acc

def evaluate_shifts():
    print("\n" + "="*60)
    print("Running Domain Shift Evaluation with fine-tuned model ...")
    print("="*60)

    # Reload best checkpoint
    backbone = CIFARResNet18(num_classes=10)
    backbone.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    model = ModelWrapper(backbone, device=DEVICE)
    model.eval()

    # Use val split (clean)
    val_tf = transforms.Compose([transforms.ToTensor()])
    full_ds = torchvision.datasets.ImageFolder(DATA_DIR, transform=val_tf)
    n_train = int(0.8 * len(full_ds))
    n_val   = len(full_ds) - n_train
    _, val_ds = random_split(full_ds, [n_train, n_val],
                              generator=torch.Generator().manual_seed(42))
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0)

    def eval_loader(loader, corruption_type=None, severity=3):
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                if corruption_type:
                    imgs = apply_domain_shift(imgs, corruption_type, severity)
                logits, _ = model(imgs)
                preds = logits.argmax(1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
        return correct / total * 100

    lines = []
    lines.append(f"Using device: {DEVICE}")
    lines.append(f"Model: CIFARResNet18 fine-tuned on cifar10/test (80% split)")
    lines.append("")

    base_acc = eval_loader(val_loader)
    lines.append(f"Clean Val Accuracy (Baseline): {base_acc:.2f}%")
    lines.append("-" * 60)
    lines.append(f"{'Corruption':<16} | {'Severity':<8} | {'Acc':>6} | {'Drop':>8}")
    lines.append("-" * 60)
    print(f"\nClean Val Accuracy (Baseline): {base_acc:.2f}%")
    print("-" * 60)
    print(f"{'Corruption':<16} | {'Severity':<8} | {'Acc':>6} | {'Drop':>8}")
    print("-" * 60)

    for shift in CORRUPTIONS_REGISTRY.keys():
        acc  = eval_loader(val_loader, corruption_type=shift, severity=3)
        drop = base_acc - acc
        line = f"{shift:<16} | {3:<8} | {acc:>5.2f}% | {drop:>+7.2f}%"
        print(line)
        lines.append(line)

    lines.append("-" * 60)

    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nResults saved to: {RESULT_FILE}")
    return base_acc, lines

if __name__ == "__main__":
    best_val = train()
    base_acc, result_lines = evaluate_shifts()
    print("\nDone! Update presentation.html with these numbers.")
