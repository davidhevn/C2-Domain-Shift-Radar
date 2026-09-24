import os
import torch
import torch.nn.functional as F
from src.data.loader import get_cifar10_dataloaders
from src.data.corruptions import apply_domain_shift, CORRUPTIONS_REGISTRY
from src.models.model import load_model

def compute_distances():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and data
    model = load_model(device=device)
    model.eval()
    
    data_dir = os.path.dirname(__file__)
    _, _, test_loader = get_cifar10_dataloaders(data_dir=data_dir, batch_size=100, val_size=100)
    
    # Lấy 1 batch duy nhất để tính toán minh họa cho lẹ
    clean_images, labels = next(iter(test_loader))
    clean_images = clean_images.to(device)
    
    # Trích xuất đặc trưng của tập Clean
    with torch.no_grad():
        clean_out = model.predict_batch(clean_images)
        clean_features = clean_out["features"]
        
        # Calculate clean accuracy just for reference
        clean_preds = clean_out["preds"]
        clean_acc = (clean_preds == labels.to(device)).float().mean().item()
        
    print(f"\nClean Baseline Accuracy: {clean_acc * 100:.2f}%")
    print("-" * 60)
    print(f"{'Corruption':<15} | {'Pixel L2 Dist':<15} | {'Feature Dist':<15} | {'Acc Drop':<10}")
    print("-" * 60)

    # Thử từng loại nhiễu
    for shift_type in CORRUPTIONS_REGISTRY.keys():
        shifted_images = apply_domain_shift(clean_images.cpu(), corruption_type=shift_type, severity=3).to(device)
        
        with torch.no_grad():
            shifted_out = model.predict_batch(shifted_images)
            shifted_features = shifted_out["features"]
            
            shifted_preds = shifted_out["preds"]
            shifted_acc = (shifted_preds == labels.to(device)).float().mean().item()
            drop = clean_acc - shifted_acc
            
        # 1. Pixel Distance (Bẫy: Khoảng cách L2 trên pixel thô)
        # Tính khoảng cách trung bình của các ảnh
        pixel_dist = F.mse_loss(shifted_images, clean_images).item()
        
        # 2. Feature Distance (Giải pháp: Khoảng cách Cosine ở không gian Embeddings)
        # 1 - cosine_similarity. Nếu giống nhau hoàn toàn = 0.
        feat_sim = F.cosine_similarity(clean_features, shifted_features, dim=1).mean().item()
        feat_dist = 1.0 - feat_sim
        
        print(f"{shift_type:<15} | {pixel_dist:<15.4f} | {feat_dist:<15.4f} | {drop*100:>8.2f}%")
        
    print("-" * 60)
    print("\n[PHÂN TÍCH 'BẪY' PRODUCTION]")
    print("=> Bẫy: Các metric nhìn 'rất khoa học' như Pixel L2 Distance thường đánh giá nhiễu Salt & Pepper")
    print("   hoặc Fog là cực kỳ nghiêm trọng (Pixel Dist rất cao). Nhưng thực tế Acc Drop lại âm (tức là không tụt).")
    print("=> Giải pháp: Feature Distance (Khoảng cách ở không gian nhúng của Model) phản ánh chuẩn xác hơn.")
    print("   Nếu Feature Dist thấp, nghĩa là mô hình đã học được 'sự bất biến' (invariance) với nhiễu đó,")
    print("   nên dù mắt thường thấy ảnh hỏng, model vẫn dự đoán đúng -> KHÔNG ĐƯỢC BÁO ĐỘNG GIẢ!")

if __name__ == "__main__":
    compute_distances()
