import os
import sys
import json
import random
import unicodedata
from collections import Counter
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Tắt cảnh báo Hugging Face và Tokenizers
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Fix Unicode error for Windows terminal
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Cấu hình thiết bị và đường dẫn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tự động phát hiện Google Colab
IN_COLAB = os.path.exists('/content')
if IN_COLAB:
    SAVE_PATH = "/content/drive/MyDrive/Temo/search/file_train"
    if not os.path.exists(SAVE_PATH): 
        try:
            os.makedirs(SAVE_PATH, exist_ok=True)
        except:
            print("⚠️ Cảnh báo: Không thể tạo thư mục trên Drive. Kiểm tra xem bạn đã Mount Drive chưa?")
            SAVE_PATH = "file_train"
    print(f">>> Đang chạy trên Colab. Kết quả sẽ lưu vào Drive: {SAVE_PATH}")
else:
    SAVE_PATH = "file_train"
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH, exist_ok=True)
    print(f">>> Đang chạy cục bộ. Kết quả sẽ lưu vào: {SAVE_PATH}")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f">>> Đã cố định hạt giống ngẫu nhiên (seed={seed}) để đảm bảo tính nhất quán.")

def clean_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = " ".join(text.split())
    return text.strip()

class MaroMartDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_input_len=128, max_target_len=64):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        input_text = "intent: " + clean_text(row['input'])
        target_text = clean_text(row['product_title'])

        inputs = self.tokenizer.encode_plus(
            input_text,
            max_length=self.max_input_len,
            padding=False,  # DO NOT pad globally
            truncation=True,
            return_tensors="pt"
        )

        targets = self.tokenizer.encode_plus(
            target_text,
            max_length=self.max_target_len,
            padding=False,  # DO NOT pad globally
            truncation=True,
            return_tensors="pt"
        )

        labels = targets["input_ids"].flatten()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": inputs["input_ids"].flatten(),
            "attention_mask": inputs["attention_mask"].flatten(),
            "labels": labels
        }

class SmartCollate:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        input_ids = [item["input_ids"] for item in batch]
        attention_masks = [item["attention_mask"] for item in batch]
        labels = [item["labels"] for item in batch]

        # Padding động theo độ dài lớn nhất của batch hiện tại
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        attention_masks = torch.nn.utils.rnn.pad_sequence(
            attention_masks, batch_first=True, padding_value=0
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "labels": labels
        }

def get_token_f1(pred, label):
    pred_tokens = str(pred).strip().lower().split()
    label_tokens = str(label).strip().lower().split()
    if len(pred_tokens) == 0 or len(label_tokens) == 0:
        return 1.0 if pred_tokens == label_tokens else 0.0
    common = Counter(pred_tokens) & Counter(label_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(label_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1

def calculate_metrics(preds_text, labels_text):
    if len(labels_text) == 0: 
        return 0.0, 0.0, 0.0, 0.0
    
    # 1. Tính Accuracy (Exact Match)
    matches = sum([1 for p, l in zip(preds_text, labels_text) if p.strip().lower() == l.strip().lower()])
    acc = matches / len(labels_text)
    
    # 2. Tính Token-level Precision, Recall, F1
    total_pre, total_rec, total_f1 = 0.0, 0.0, 0.0
    for p, l in zip(preds_text, labels_text):
        pre, rec, f1 = get_token_f1(p, l)
        total_pre += pre
        total_rec += rec
        total_f1 += f1
        
    avg_pre = total_pre / len(labels_text)
    avg_rec = total_rec / len(labels_text)
    avg_f1 = total_f1 / len(labels_text)
    
    return acc, avg_pre, avg_rec, avg_f1

def train_and_evaluate(config, train_df, val_df):
    set_seed(42)
    model_name = "VietAI/vit5-base"
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    
    # Cấu hình Dropout trực tiếp khi load model để đảm bảo tính nhất quán
    model = T5ForConditionalGeneration.from_pretrained(model_name, dropout_rate=config['dropout'])
    model.to(device)

    train_dataset = MaroMartDataset(train_df, tokenizer)
    val_dataset = MaroMartDataset(val_df, tokenizer)

    collate_fn = SmartCollate(tokenizer.pad_token_id)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        collate_fn=collate_fn,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        collate_fn=collate_fn,
        pin_memory=True
    )

    optimizer = AdamW(model.parameters(), lr=config['lr'], weight_decay=0.01)
    
    epochs = 40
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    scaler = torch.cuda.amp.GradScaler()
    
    # Đánh giá độc lập trên validation
    def evaluate_model(loader):
        model.eval()
        all_preds_text = []
        all_labels_text = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"]
                
                # AI sinh câu trả lời trong mixed precision tự động tăng tốc và tối ưu VRAM
                with torch.cuda.amp.autocast():
                    outputs = model.generate(
                        input_ids=input_ids, 
                        attention_mask=attention_mask, 
                        max_new_tokens=48,
                        early_stopping=True
                    )
                
                preds_batch = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                
                labels_ids = labels.clone()
                labels_ids[labels_ids == -100] = tokenizer.pad_token_id
                labels_batch = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
                
                all_preds_text.extend(preds_batch)
                all_labels_text.extend(labels_batch)
        
        # Dọn dẹp bộ nhớ đệm CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return calculate_metrics(all_preds_text, all_labels_text)

    best_val_f1 = -1.0
    patience = 6
    epochs_no_improve = 0
    model_path = f"{SAVE_PATH}/best_model"

    print(">>> Bắt đầu huấn luyện...")
    for epoch in range(epochs):
        model.train()
        train_bar = tqdm(train_loader, desc=f"   Epoch {epoch+1}/{epochs}", leave=False)
        total_loss = 0.0
        
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
            
            # Gradient clipping để tránh bùng nổ độ dốc
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")
            
        avg_loss = total_loss / len(train_loader)
        
        # Đánh giá hiệu năng Validation sau mỗi epoch để phát hiện Best Checkpoint
        val_acc, val_pre, val_rec, val_f1 = evaluate_model(val_loader)
        print(f"   Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | Val F1: {val_f1:.4f} | Val Acc (EM): {val_acc:.4f}")
        
        # Lưu model tốt nhất
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
            print(f"   🌟 Đã cập nhật Checkpoint tốt nhất tại Epoch {epoch+1} với Val F1 = {best_val_f1:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"   🛑 Dừng sớm tại Epoch {epoch+1} vì điểm Val F1 không cải thiện sau {patience} epochs.")
                break

    # Load lại model tốt nhất đã lưu để kiểm định cuối cùng
    print(f">>> Huấn luyện hoàn tất! Nạp lại model tốt nhất từ {model_path} để kiểm định cuối cùng...")
    best_model = T5ForConditionalGeneration.from_pretrained(model_path)
    best_model.to(device)
    
    # Đánh giá cuối cùng trên cả Train và Val
    train_acc, train_pre, train_rec, train_f1 = evaluate_model(train_loader)
    val_acc, val_pre, val_rec, val_f1 = evaluate_model(val_loader)

    return {
        "train": {"accuracy": train_acc, "precision": train_pre, "recall": train_rec, "f1": train_f1},
        "val": {"accuracy": val_acc, "precision": val_pre, "recall": val_rec, "f1": val_f1}
    }, best_model

def main():
    print(">>> Bắt đầu chuẩn bị dữ liệu...")
    dataset_file = "data/augmented_dataset.csv"
    if not os.path.exists(dataset_file):
        print(f"❌ Không tìm thấy tệp {dataset_file}. Vui lòng chạy sinh dữ liệu trước!")
        return

    df = pd.read_csv(dataset_file)
    
    # Tiền xử lý dữ liệu trước khi train
    df['input'] = df['input'].apply(clean_text)
    df['product_title'] = df['product_title'].apply(clean_text)
    df = df.dropna(subset=['input', 'product_title'])
    df = df[(df['input'] != "") & (df['product_title'] != "")]
    
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f">>> Phân phối dữ liệu: Train = {len(train_df)} dòng | Val = {len(val_df)} dòng.")

    # Siêu tham số tối ưu đã được thiết lập trực tiếp để rút ngắn thời gian và tối ưu VRAM
    config = {'dropout': 0.1, 'batch_size': 16, 'lr': 5e-5}

    print(f"--- 🛠️ Cấu hình huấn luyện: {config} ---")
    scores, model = train_and_evaluate(config, train_df, val_df)
    
    # In báo cáo kết quả chi tiết theo yêu cầu của bạn
    print("\n" + "="*50)
    print("🏆 BÁO CÁO KẾT QUẢ HUẤN LUYỆN CUỐI CÙNG (FINAL METRICS)")
    print("="*50)
    print("Tập huấn luyện (Training Set):")
    print(f"  - Accuracy (Exact Match): {scores['train']['accuracy']*100:.2f}%")
    print(f"  - Precision (Token):      {scores['train']['precision']*100:.2f}%")
    print(f"  - Recall (Token):         {scores['train']['recall']*100:.2f}%")
    print(f"  - F1-Score (Token):       {scores['train']['f1']*100:.2f}%")
    print("-"*50)
    print("Tập kiểm thử (Validation Set):")
    print(f"  - Accuracy (Exact Match): {scores['val']['accuracy']*100:.2f}%")
    print(f"  - Precision (Token):      {scores['val']['precision']*100:.2f}%")
    print(f"  - Recall (Token):         {scores['val']['recall']*100:.2f}%")
    print(f"  - F1-Score (Token):       {scores['val']['f1']*100:.2f}%")
    print("="*50 + "\n")

    # Ghi nhận kết quả huấn luyện vào tệp JSON
    result_file = f"{SAVE_PATH}/vit5_training_results.json"
    results = [{
        "config": config,
        "metrics": scores
    }]
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"✅ Đã cập nhật báo cáo và lưu kết quả vào: {result_file}")

if __name__ == "__main__":
    main()
