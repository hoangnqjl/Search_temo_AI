import os
import sys
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import json
import numpy as np
from tqdm import tqdm

# Tắt cảnh báo Hugging Face và Tokenizers
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Fix Unicode error for Windows terminal
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Cấu hình thiết bị và đường dẫn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tự động phát hiện Google Colab (kiểm tra đường dẫn hệ thống)
IN_COLAB = os.path.exists('/content')
if IN_COLAB:
    # Đảm bảo đường dẫn lưu vào Drive
    SAVE_PATH = "/content/drive/MyDrive/Temo/search/file_train"
    if not os.path.exists(SAVE_PATH): 
        try:
            os.makedirs(SAVE_PATH, exist_ok=True)
        except:
            print("⚠️ Cảnh báo: Không thể tạo thư mục trên Drive. Kiểm tra xem bạn đã Mount Drive chưa?")
            SAVE_PATH = "data"
    print(f">>> Đang chạy trên Colab. Kết quả sẽ lưu vào Drive: {SAVE_PATH}")
else:
    SAVE_PATH = "data"
    print(f">>> Đang chạy cục bộ. Kết quả sẽ lưu vào: {SAVE_PATH}")

class MaroMartDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_input_len=128, max_target_len=256):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        input_text = "intent: " + str(row['input'])
        target_text = str(row['product_title']) + " - " + str(row['product_description'])

        inputs = self.tokenizer.encode_plus(
            input_text,
            max_length=self.max_input_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        targets = self.tokenizer.encode_plus(
            target_text,
            max_length=self.max_target_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        labels = targets["input_ids"].flatten()
        # Sửa lỗi quan trọng: Bỏ qua padding khi tính Loss để model không bị "câm"
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": inputs["input_ids"].flatten(),
            "attention_mask": inputs["attention_mask"].flatten(),
            "labels": labels
        }

def calculate_metrics(preds, labels):
    # Đưa về dạng 1D để tính metrics
    preds = preds.flatten()
    labels = labels.flatten()
    
    # Lọc bỏ phần padding (thường là 0)
    mask = labels != 0
    preds = preds[mask]
    labels = labels[mask]
    
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    
    return acc, precision, recall, f1

def train_and_evaluate(config, train_df, val_df):
    model_name = "VietAI/vit5-base"
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # Cấu hình Dropout
    model.config.dropout_rate = config['dropout']
    model.to(device)

    train_dataset = MaroMartDataset(train_df, tokenizer)
    val_dataset = MaroMartDataset(val_df, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])

    optimizer = AdamW(model.parameters(), lr=config['lr'])
    scaler = torch.cuda.amp.GradScaler()
    
    # Training Loop (đơn giản hóa cho Grid Search)
    epochs = 3
    for epoch in range(epochs):
        model.train()
        train_bar = tqdm(train_loader, desc=f"   Epoch {epoch+1}/{epochs}", leave=False)
        for batch in train_bar:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Sử dụng Mixed Precision (FP16) để tiết kiệm VRAM
            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

    # Evaluation
    model.eval()
    def get_scores(loader, desc="Evaluating"):
        all_preds = []
        all_labels = []
        with torch.no_grad():
            eval_bar = tqdm(loader, desc=f"   {desc}", leave=False)
            for batch in eval_bar:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                outputs = model.generate(input_ids=input_ids, max_length=256)
                
                # Padding outputs to match labels length for metric calculation
                padded_outputs = torch.zeros_like(labels)
                limit = min(outputs.shape[1], labels.shape[1])
                padded_outputs[:, :limit] = outputs[:, :limit]
                
                all_preds.append(padded_outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        return calculate_metrics(np.concatenate(all_preds), np.concatenate(all_labels))

    train_acc, train_pre, train_rec, train_f1 = get_scores(train_loader, "Eval Train")
    val_acc, val_pre, val_rec, val_f1 = get_scores(val_loader, "Eval Val")

    return {
        "train": {"accuracy": train_acc, "precision": train_pre, "recall": train_rec, "f1": train_f1},
        "val": {"accuracy": val_acc, "precision": val_pre, "recall": val_rec, "f1": val_f1}
    }, model

def main():
    print(">>> Bắt đầu chuẩn bị dữ liệu...")
    df = pd.read_csv("data/semantic_dataset.csv")
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # Siêu tham số cho Grid Search
    # Rút gọn: Chạy 1 tổ hợp tốt nhất để tiết kiệm thời gian
    dropouts = [0.1]
    batch_sizes = [16]
    learning_rates = [3e-5]

    # Đọc kết quả cũ từ Drive để hỗ trợ chạy tiếp (Resume)
    result_file = f"{SAVE_PATH}/vit5_training_results.json"
    results = []
    best_f1 = -1.0
    completed_configs = []

    if os.path.exists(result_file):
        try:
            with open(result_file, "r", encoding="utf-8") as f:
                results = json.load(f)
            for r in results:
                completed_configs.append(r['config'])
                f1 = r['metrics']['val']['f1']
                if f1 > best_f1: best_f1 = f1
            print(f">>> 📂 Tìm thấy {len(results)} tổ hợp đã chạy xong. Sẽ tự động bỏ qua.")
            print(f">>> 🏆 Điểm F1 tốt nhất hiện tại: {best_f1:.4f}")
        except Exception as e:
            print(f"⚠️ Cảnh báo: Không thể đọc file kết quả cũ: {e}")

    total_configs = len(dropouts) * len(batch_sizes) * len(learning_rates)
    count = 0

    print(f">>> 🚀 Bắt đầu Grid Search với {total_configs} tổ hợp...")

    for d in dropouts:
        for b in batch_sizes:
            for lr in learning_rates:
                count += 1
                config = {'dropout': d, 'batch_size': b, 'lr': lr}
                
                # Kiểm tra xem tổ hợp này đã chạy chưa
                if config in completed_configs:
                    print(f"--- ⏩ [{count}/{total_configs}] Bỏ qua: {config} (Đã có kết quả) ---")
                    continue

                print(f"--- 🛠️ [{count}/{total_configs}] Đang huấn luyện: {config} ---")
                
                scores, model = train_and_evaluate(config, train_df, val_df)
                
                # Kiểm tra và lưu model tốt nhất
                current_f1 = scores['val']['f1']
                if current_f1 > best_f1:
                    best_f1 = current_f1
                    model_path = f"{SAVE_PATH}/best_model"
                    model.save_pretrained(model_path)
                    # Load lại tokenizer để lưu cùng model
                    from transformers import T5Tokenizer
                    tokenizer = T5Tokenizer.from_pretrained("VietAI/vit5-base", legacy=False)
                    tokenizer.save_pretrained(model_path)
                    print(f"🌟 MỚI: Đã tìm thấy model tốt nhất với F1={best_f1:.4f}. Đã lưu vào {model_path}")

                results.append({
                    "config": config,
                    "metrics": scores
                })

                # Lưu kết quả liên tục sau mỗi tổ hợp để tránh mất dữ liệu
                result_file = f"{SAVE_PATH}/vit5_training_results.json"
                with open(result_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)
                
                print(f"--- 💾 Đã lưu kết quả tổ hợp {count} vào Drive ---")

    print(f"\n✅ Hoàn tất Grid Search! Toàn bộ kết quả đã ở tại {SAVE_PATH}")

if __name__ == "__main__":
    main()
