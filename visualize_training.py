import json
import os
import matplotlib.pyplot as plt

def visualize_training():
    results_path = "file_train/vit5_training_results.json"
    
    if not os.path.exists(results_path):
        print(f"❌ Không tìm thấy file {results_path}!")
        print("Vui lòng chạy lại file maromart_vit5_training.ipynb trên Colab để sinh ra file kết quả mới.")
        return
        
    with open(results_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            # Dữ liệu có thể là list hoặc dict tùy vào cách dump
            if isinstance(data, list) and len(data) > 0:
                data = data[0]
        except Exception as e:
            print(f"Lỗi đọc file JSON: {e}")
            return
            
    if "history" not in data:
        print("❌ File kết quả hiện tại thuộc phiên bản cũ (không có dữ liệu history từng epoch).")
        print("Vui lòng tải file maromart_vit5_training.ipynb mới cập nhật lên Colab và Train lại!")
        return
        
    history = data["history"]
    if not history:
        print("Dữ liệu history rỗng.")
        return
        
    epochs = [item["epoch"] for item in history]
    losses = [item["loss"] for item in history]
    val_f1s = [item["val_f1"] for item in history]
    
    # Tạo hình ảnh gồm 2 biểu đồ (Loss và F1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Biểu đồ 1: Loss
    ax1.plot(epochs, losses, 'b-', marker='o', linewidth=2, markersize=6, label='Training Loss')
    ax1.set_title('ViT5 Intent Extraction - Training Loss', fontsize=14, pad=15)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    # Biểu đồ 2: F1-Score
    ax2.plot(epochs, val_f1s, 'g-', marker='s', linewidth=2, markersize=6, label='Validation F1')
    ax2.set_title('ViT5 Intent Extraction - Validation F1 Score', fontsize=14, pad=15)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('F1 Score', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Highlight max F1
    max_f1 = max(val_f1s)
    max_epoch = epochs[val_f1s.index(max_f1)]
    ax2.annotate(f'Max F1: {max_f1*100:.2f}% (Epoch {max_epoch})', 
                 xy=(max_epoch, max_f1), 
                 xytext=(max_epoch, max_f1 - 0.1),
                 arrowprops=dict(facecolor='red', shrink=0.05),
                 fontsize=11, color='red',
                 ha='center')
    ax2.legend()
    
    plt.tight_layout()
    
    # Lưu file ảnh
    save_path = "file_train/vit5_training_visualization.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Đã lưu biểu đồ Visualize quá trình mất mát và hội tụ tại: {save_path}")
    
    # Hiển thị
    plt.show()

if __name__ == "__main__":
    visualize_training()
