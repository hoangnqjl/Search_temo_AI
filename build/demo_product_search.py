import pandas as pd
import torch
import sys
from transformers import AutoModel, AutoTokenizer
import numpy as np
import os

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

# 5. Hàm tìm kiếm dựa trên độ tương đồng Cosine
def semantic_search(query, top_n=5):
    query_vec = get_embedding(query)
    
    # Tính toán Cosine Similarity
    norm_products = np.linalg.norm(product_embeddings, axis=1)
    norm_query = np.linalg.norm(query_vec)
    similarities = np.dot(product_embeddings, query_vec.T).flatten() / (norm_products * norm_query)
    
    # Lấy top N kết quả cao nhất
    top_indices = similarities.argsort()[-top_n:][::-1]
    return df.iloc[top_indices], similarities[top_indices]

# 6. Giao diện Demo tương tác
print("\n" + "="*50)
print("🌟 CHÀO MỪNG ĐẾN VỚI HỆ THỐNG TÌM KIẾM NGỮ NGHĨA MAROMART 🌟")
print("="*50)
print("Gợi ý: Bạn có thể nhập câu hỏi dài, mô tả nhu cầu thay vì chỉ gõ từ khóa.")

while True:
    user_query = input("\n🔍 Bạn đang tìm kiếm gì? (Gõ 'exit' để thoát): ")
    if user_query.lower() in ['exit', 'quit', 'thoát']:
        print("Tạm biệt!")
        break
    
    results, scores = semantic_search(user_query)
    
    print(f"\n--- Top {len(results)} sản phẩm phù hợp nhất cho: '{user_query}' ---")
    for i, (idx, row) in enumerate(results.iterrows()):
        score_percent = scores[i] * 100
        print(f"{i+1}. [{row['productName']}]")
        print(f"   💰 Giá: {row['productPrice']:,}đ | 🏷️ Thương hiệu: {row['productBrand']}")
        print(f"   📈 Độ khớp: {score_percent:.2f}%")
        print(f"   📝 Mô tả ngắn: {row['productDescription'][:120]}...")
        print("-" * 30)
