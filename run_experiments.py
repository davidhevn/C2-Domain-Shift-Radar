import torch
from src.data.loader import get_cifar10_dataloaders
from src.models.model import load_model
from src.data.corruptions import apply_domain_shift, CORRUPTIONS_REGISTRY

def evaluate(model_wrapper, dataloader, device, corruption_type=None, severity=1):
    model_wrapper.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            if corruption_type is not None:
                inputs = apply_domain_shift(inputs, corruption_type, severity)
            
            outputs = model_wrapper.predict_batch(inputs)
            preds = outputs['preds']
            
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
    return (correct / total) * 100

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading data...")
    train_loader, val_loader, test_loader = get_cifar10_dataloaders(data_dir="./data", batch_size=256)
    
    print("Loading model...")
    model_wrapper = load_model(device=device)
    
    print("Evaluating on Original Clean Test Set...")
    base_acc = evaluate(model_wrapper, test_loader, device)
    print(f"Original Test Accuracy: {base_acc:.2f}%")
    print("-" * 50)
    
    print("Evaluating on Domain Shifts (Severity=3)...")
    results = {}
    
    corruptions_to_test = list(CORRUPTIONS_REGISTRY.keys())
    severity = 3
    
    for corr in corruptions_to_test:
        acc = evaluate(model_wrapper, test_loader, device, corruption_type=corr, severity=severity)
        drop = base_acc - acc
        results[corr] = {'acc': acc, 'drop': drop}
        print(f"Corruption: {corr:<15} | Severity: {severity} | Acc: {acc:>5.2f}% | Drop: {drop:>5.2f}%")
        
if __name__ == "__main__":
    main()
