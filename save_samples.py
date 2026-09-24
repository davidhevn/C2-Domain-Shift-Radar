import os
import torch
import torchvision
from src.data.loader import get_cifar10_dataloaders
from src.data.corruptions import apply_domain_shift, CORRUPTIONS_REGISTRY

def main():
    save_dir = os.path.join("cifar10", "shifted_samples")
    os.makedirs(save_dir, exist_ok=True)
    
    train_loader, val_loader, test_loader = get_cifar10_dataloaders(data_dir=".", batch_size=1)
    
    # Lấy thử 1 tấm ảnh đầu tiên trong tập test
    inputs, targets = next(iter(test_loader))
    
    # Lưu ảnh gốc
    torchvision.utils.save_image(inputs, os.path.join(save_dir, "0_original.png"))
    
    # Lưu các ảnh đã bị làm nhiễu
    severity = 3
    for corr in CORRUPTIONS_REGISTRY.keys():
        shifted_img = apply_domain_shift(inputs.clone(), corr, severity)
        filename = f"{corr}_severity{severity}.png"
        torchvision.utils.save_image(shifted_img, os.path.join(save_dir, filename))
        
    print(f"Đã lưu các ảnh mẫu vào thư mục: {os.path.abspath(save_dir)}")

if __name__ == "__main__":
    main()
