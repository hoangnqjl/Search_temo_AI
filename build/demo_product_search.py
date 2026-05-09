import pandas as pd
import torch
import sys
from transformers import AutoModel, AutoTokenizer
import numpy as np
import os
import re

# --- CẤU HÌNH ---
IN_COLAB = 'google.colab' in sys.modules
# Sử dụng model chuyên biệt cho tìm kiếm tiếng Việt để đạt độ chính xác cao nhất
MODEL_NAME = "keepitreal/vietnamese-sbert" 
PRODUCT_CSV = "data/products.csv"

# Tự động quét model trên Drive nếu đang ở Colab
if IN_COLAB:
    drive_model = "/content/drive/MyDrive/Temo/search/file_train/best_model"
    if os.path.exists(drive_model):
        MODEL_NAME = drive_model
        print(f">>> Đã tìm thấy model trên Drive: {MODEL_NAME}")

print(f">>> Đang khởi tạo hệ thống tìm kiếm SBERT: {MODEL_NAME}...")

# 1. Load Model và Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

# 2. Load dữ liệu sản phẩm thực tế
if not os.path.exists(PRODUCT_CSV):
    print(f"❌ Không tìm thấy file {PRODUCT_CSV}. Vui lòng chạy export_csv trước.")
    exit()

df = pd.read_csv(PRODUCT_CSV)
df = df.fillna("") # Xử lý các ô trống

# 3. Hàm tạo Vector đặc trưng (Embedding) bằng SBERT
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    with torch.no_grad():
        # Lấy hidden state của token đầu tiên [CLS] làm vector đại diện cho câu
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
    return embeddings.numpy()

# 4. Mã hóa toàn bộ sản phẩm (Nên làm một lần và lưu lại)
print(f">>> Đang mã hóa {len(df)} sản phẩm. Vui lòng đợi trong giây lát...")
product_texts = (df['productName'] + " " + df['productDescription'] + " " + df['categoryName']).tolist()
product_embeddings = []

for text in product_texts:
    product_embeddings.append(get_embedding(text))

product_embeddings = np.vstack(product_embeddings)

# 5. Hàm bóc tách ngân sách từ câu hỏi (Budget Intelligence)
def extract_budget(query):
    query = query.lower()
    # Regex tìm số + đơn vị (triệu, tr, k, đồng)
    patterns = [
        (r'(\d+)\s*triệu', 1000000),
        (r'(\d+)\s*tr', 1000000),
        (r'(\d+)\s*k', 1000),
        (r'(\d+)\s*ngàn', 1000),
        (r'(\d+)\s*đ', 1),
        (r'(\d+)\s*đồng', 1),
        (r'(\d+)\.(\d+)', 1), # Trường hợp 5.000.000
    ]
    
    for pattern, multiplier in patterns:
        match = re.search(pattern, query)
        if match:
            try:
                val = float(match.group(1).replace(".", ""))
                return val * multiplier
            except: continue
    return None

# 6. Hàm lọc ý định bằng ViT5 (Giúp tìm kiếm chính xác hơn)
def refine_query(query):
    # Đường dẫn tới model ViT5 đã train trên Drive
    vit5_path = "/content/drive/MyDrive/Temo/search/file_train/best_model"
    if not os.path.exists(vit5_path):
        return query
    
    try:
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        v_tokenizer = T5Tokenizer.from_pretrained("VietAI/vit5-base", legacy=False)
        v_model = T5ForConditionalGeneration.from_pretrained(vit5_path)
        v_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        input_text = "intent: " + query
        inputs = v_tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(v_model.device)
        
        with torch.no_grad():
            outputs = v_model.generate(
                **inputs, max_length=30, num_beams=5, 
                repetition_penalty=2.5, no_repeat_ngram_size=3, early_stopping=True
            )
            refined = v_tokenizer.decode(outputs[0], skip_special_tokens=True)
            if refined.strip():
                refined = refined.replace("*", "").strip()
                return refined
        return query
    except:
        return query

# 7. Hàm tìm kiếm dựa trên độ tương đồng Cosine + Ngân sách
def semantic_search(query, top_n=5):
    # Bước 1: Bóc tách ngân sách mong muốn
    budget = extract_budget(query)
    if budget:
        print(f"💰 Hệ thống ghi nhận ngân sách: {budget:,}đ")
    
    # Bước 2: Dùng ViT5 để "lọc" ý định
    clean_query = refine_query(query)
    if clean_query != query:
        print(f"✨ AI hiểu nhu cầu: '{clean_query}'")
    
    # Bước 3: Dùng SBERT lấy vector
    query_vec = get_embedding(clean_query)
    
    # Bước 4: Tính toán điểm số
    norm_products = np.linalg.norm(product_embeddings, axis=1)
    norm_query = np.linalg.norm(query_vec)
    similarities = np.dot(product_embeddings, query_vec.T).flatten() / (norm_products * norm_query)
    
    # Bước 5: Phạt các sản phẩm vượt quá ngân sách (nếu có)
    if budget:
        for i, row in df.iterrows():
            price = float(row['productPrice'])
            # Nếu giá vượt quá ngân sách, giảm điểm độ khớp
            if price > budget * 1.2: # Cho phép lệch 20%
                penalty = budget / price # Giá càng cao phạt càng nặng
                similarities[i] *= penalty
    
    # Lấy top N kết quả
    top_indices = similarities.argsort()[-top_n:][::-1]
    return df.iloc[top_indices], similarities[top_indices], clean_query

# 8. Giao diện Demo tương tác
print("\n" + "="*50)
print("🌟 CHÀO MỪNG ĐẾN VỚI HỆ THỐNG TÌM KIẾM THÔNG MINH MAROMART 🌟")
print("="*50)
print("Gợi ý: Bạn có thể nhập nhu cầu và ngân sách (VD: iPhone dưới 10 triệu)")

while True:
    user_query = input("\n🔍 Bạn đang tìm kiếm gì? (Gõ 'exit' để thoát): ")
    if user_query.lower() in ['exit', 'quit', 'thoát']:
        break
    
    results, scores, clean_q = semantic_search(user_query)
    
    print(f"\n--- Kết quả tìm kiếm cho: '{user_query}' ---")
    for i, (idx, row) in enumerate(results.iterrows()):
        score_percent = scores[i] * 100
        print(f"{i+1}. [{row['productName']}]")
        print(f"   💰 Giá: {row['productPrice']:,}đ | 🏷️ Thương hiệu: {row['productBrand']}")
        print(f"   📈 Độ khớp (đã tính giá): {score_percent:.2f}%")
        print(f"   📝 Mô tả ngắn: {str(row['productDescription'])[:120]}...")
        print("-" * 30)
