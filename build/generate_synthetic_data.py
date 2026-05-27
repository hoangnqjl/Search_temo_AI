import os
import sys
import random
import pandas as pd
import numpy as np

# Fix Unicode error for Windows terminal printing
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def clean_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    import unicodedata
    text = unicodedata.normalize("NFC", text)
    text = " ".join(text.split())
    return text.strip()

def generate_queries():
    print(">>> Đang chạy kịch bản sinh dữ liệu tự động...")
    
    # 1. Đọc dữ liệu gốc
    dataset_path = "data/semantic_dataset.csv"
    if not os.path.exists(dataset_path):
        print(f"❌ Không tìm thấy file {dataset_path}!")
        return
        
    df_orig = pd.read_csv(dataset_path)
    print(f">>> Đã đọc dữ liệu gốc: {len(df_orig)} dòng.")
    
    # Lấy danh sách sản phẩm độc bản từ dữ liệu gốc
    products = df_orig[['product_title', 'product_description', 'is_new']].drop_duplicates().to_dict('records')
    print(f">>> Tìm thấy {len(products)} sản phẩm độc bản để sinh dữ liệu.")
    
    # 2. Xây dựng bộ từ vựng và cấu trúc hội thoại đời thường của người Việt
    xung_ho_dau = [
        "em", "mình", "tớ", "anh", "chị", "khách", "nhà em", "bé nhà mình", 
        "con em", "bố mẹ em", "vợ chồng em", "cu nhà em", "bé nhà em", ""
    ]
    
    xung_ho_cuoi = [
        "nha shop", "ạ", "nhé", "nha", "giúp em với", "với ạ", "ơi", 
        "shop ơi", "ad ơi", "giúp mình nha", "cho e xin giá", "được không shop", 
        "được không ạ", "inbox em nha", "hén", ""
    ]
    
    # Động từ tìm kiếm/mua sắm đời thường
    dong_tu = [
        "đang cần tìm mua", "cần tìm một chiếc", "đang muốn kiếm", "muốn mua thanh lý", 
        "đang cần mua gấp", "muốn sắm", "cần tìm mua lại", "muốn kiếm một bé", 
        "có bán không", "có em nào pass lại", "tìm mua", "cần sắm", "đang kiếm", 
        "muốn tìm", "kiếm giùm em", "tìm giúp mình", "hỏi thăm xem có", "cần tìm mua"
    ]
    
    # Từ lóng và viết tắt về tình trạng
    tinh_trang_cu = [
        "cũ", "2hand", "secondhand", "used", "qua sử dụng", "like new", "mới 99%", 
        "ngoại hình đẹp", "pin trâu", "còn nguyên zin", "hàng lướt", "dùng tốt", "đẹp ken"
    ]
    
    tinh_trang_moi = [
        "mới 100%", "nguyên seal", "chính hãng", "mới tinh", "đập hộp", "hàng new", "mới cứng"
    ]
    
    # Ngữ cảnh/Mục đích sử dụng tự nhiên của con người
    muc_dich_gia_dinh = [
        "về dùng cho cả nhà", "để phòng khách nhìn cho sang", "về giặt đồ cho con đỡ mùi mốc",
        "cho người già bị tiểu đường ở nhà ăn", "để phòng bếp cho tiện ăn cơm gia đình",
        "cho bé 2 tuổi phát triển tư duy hình khối", "lưu giữ khoảnh khắc kỷ niệm đẹp cho gia đình",
        "để sắp xếp lại đống giày dép ở cửa ra vào", "để nằm mát lạnh không bị đau lưng",
        "để trang trí bàn làm việc cho đẹp", "về cho bé tập tô vẽ đỡ nghịch điện thoại"
    ]
    
    muc_dich_hoc_lam = [
        "để đi học đại học", "đi làm văn phòng hàng ngày", "để chạy mượt Photoshop với Illustrator",
        "cho con gái đi học cấp 3 đỡ mỏi chân", "cho con tập chơi guitar bấm nhẹ không đau tay",
        "để in đơn hàng từ điện thoại cho shop online", "về tự tập làm móng mỗi cuối tuần cho tiết kiệm",
        "cho con xem youtube với học tiếng anh online", "để cài lại win với phần mềm văn phòng tận nhà",
        "để sạc siêu tốc cho điện thoại đi phượt", "để mang đi làm hàng ngày đựng đồ cá nhân"
    ]
    
    muc_dich_choi_phuot = [
        "đi dã ngoại cuối tuần cùng công ty hóng gió biển", "để mang đi phượt bền bỉ chống va đập",
        "đi chơi tennis cho người mới tập không bị đuối", "đi chạy bộ êm ái bảo vệ cổ chân",
        "đi phượt sạc nhanh cho cả iPhone và Samsung", "cho chuyến đi du lịch biển sắp tới chụp hình cho xinh"
    ]

    # Các từ lóng về tiền tệ
    don_vi_trieu = ["tr", "củ", "triệu", "tỏi"]
    don_vi_nghin = ["k", "ngàn", "nghìn"]

    generated_rows = []
    
    # Mục tiêu sinh khoảng 3000+ dòng
    # Với mỗi sản phẩm độc bản, ta sẽ sinh khoảng 15-20 câu truy vấn mô tả nhu cầu khác nhau
    num_variants_per_product = 16
    
    for prod in products:
        title = prod['product_title']
        desc = prod['product_description']
        is_new = prod['is_new']
        
        # Phân tích sơ bộ sản phẩm để chọn từ khóa tự nhiên phù hợp
        clean_title = clean_text(title)
        lower_title = clean_title.lower()
        
        # Chọn từ viết tắt của sản phẩm
        abbreviation = clean_title
        if "điện thoại" in lower_title or "iphone" in lower_title or "samsung" in lower_title:
            abbreviation = random.choice(["đt", "điện thoại", "máy", "con máy", "con đt"])
        elif "laptop" in lower_title or "dell" in lower_title or "lenovo" in lower_title or "asus" in lower_title or "acer" in lower_title:
            abbreviation = random.choice(["lap", "laptop", "máy", "con lap", "máy tính"])
        elif "vợt" in lower_title:
            abbreviation = random.choice(["cây vợt", "vợt", "vợt cầu lông"])
        elif "xe máy" in lower_title or "honda" in lower_title or "vision" in lower_title:
            abbreviation = random.choice(["xe máy", "xe", "con xe", "vision"])
        elif "nệm" in lower_title:
            abbreviation = random.choice(["nệm", "đệm", "tấm nệm"])
        elif "tủ lạnh" in lower_title:
            abbreviation = random.choice(["tủ lạnh", "cái tủ lạnh", "tủ mini"])
        elif "máy giặt" in lower_title:
            abbreviation = random.choice(["máy giặt", "cái máy giặt", "máy giặt cũ"])
        elif "bàn" in lower_title:
            abbreviation = random.choice(["bàn", "bàn làm việc", "bàn ăn"])
        elif "vách" in lower_title or "váy" in lower_title or "áo" in lower_title:
            abbreviation = random.choice(["váy", "đầm", "áo", "bộ đồ"])
        elif "tai nghe" in lower_title:
            abbreviation = random.choice(["tai nghe", "phone", "headphone"])
        elif "loa" in lower_title:
            abbreviation = random.choice(["loa", "cục loa", "loa bluetooth"])
            
        for _ in range(num_variants_per_product):
            # Tạo ngẫu nhiên một câu mô tả nhu cầu
            sh_dau = random.choice(xung_ho_dau)
            sh_cuoi = random.choice(xung_ho_cuoi)
            dt = random.choice(dong_tu)
            
            # Chọn tình trạng
            if is_new == 'True' or is_new is True:
                tt = random.choice(tinh_trang_moi)
            else:
                tt = random.choice(tinh_trang_cu)
                
            # Chọn ngữ cảnh ngẫu nhiên
            ctx = random.choice(muc_dich_gia_dinh + muc_dich_hoc_lam + muc_dich_choi_phuot)
            
            # Tạo ngân sách ngẫu nhiên giả định
            price_val = random.randint(1, 30)
            if price_val > 10:
                budget_str = f"{random.choice(['tầm', 'dưới', 'khoảng'])} {price_val} {random.choice(don_vi_trieu)}"
            else:
                budget_str = f"{random.choice(['tầm', 'khoảng', 'dưới'])} {price_val * 100} {random.choice(don_vi_nghin)}"
                
            # Trộn các thành phần thành câu văn hoàn chỉnh siêu con người
            sentence_structure = random.randint(1, 5)
            
            if sentence_structure == 1:
                # Cấu trúc 1: [Xưng hô đầu] [Động từ] [Sản phẩm] [Tình trạng] [Mục đích] [Ngân sách] [Xưng hô cuối]
                parts = [sh_dau, dt, abbreviation, tt, ctx, budget_str, sh_cuoi]
            elif sentence_structure == 2:
                # Cấu trúc 2: [Mục đích], [Xưng hô đầu] [Động từ] [Sản phẩm] [Ngân sách] [Tình trạng] [Xưng hô cuối]
                parts = [ctx, sh_dau, dt, abbreviation, budget_str, tt, sh_cuoi]
            elif sentence_structure == 3:
                # Cấu trúc 3: [Xưng hô đầu] [Động từ] [Sản phẩm] [Tình trạng] [Ngân sách] [Xưng hô cuối]
                parts = [sh_dau, dt, abbreviation, tt, budget_str, sh_cuoi]
            elif sentence_structure == 4:
                # Cấu trúc 4: Có ai pass lại [Sản phẩm] [Tình trạng] [Mục đích] [Ngân sách] không [Xưng hô cuối]
                parts = ["có ai pass lại", abbreviation, tt, ctx, budget_str, "không", sh_cuoi]
            else:
                # Cấu trúc 5: [Sản phẩm] [Tình trạng] [Ngân sách] [Mục đích] [Xưng hô cuối]
                parts = [abbreviation, tt, budget_str, ctx, sh_cuoi]
                
            # Làm sạch câu
            clean_parts = [p.strip() for p in parts if p and p.strip() != ""]
            raw_input = " ".join(clean_parts)
            
            # Chỉnh sửa chữ hoa đầu câu và dấu câu nhẹ nhàng cho tự nhiên
            raw_input = clean_text(raw_input)
            if raw_input:
                raw_input = raw_input[0].upper() + raw_input[1:]
                
            generated_rows.append({
                "input": raw_input,
                "product_title": title,
                "product_description": desc,
                "is_new": is_new
            })
            
    # Thêm nhiễu ngẫu nhiên hoặc biến thể để nâng cao chất lượng dữ liệu
    # Trộn thêm 1000 dòng biến thể ngắn hơn (dạng gõ tắt ngắn)
    for prod in products:
        title = prod['product_title']
        desc = prod['product_description']
        is_new = prod['is_new']
        clean_title = clean_text(title)
        
        for _ in range(5):
            # Từ khóa cực kỳ ngắn ngẫu nhiên
            words = clean_title.split()
            if len(words) > 1:
                # Lấy ngẫu nhiên vài từ
                short_keyword = " ".join(random.sample(words, random.randint(1, len(words))))
            else:
                short_keyword = clean_title
                
            # Ghép thêm từ lóng
            keyword_variants = [
                short_keyword.lower(),
                f"{short_keyword.lower()} {random.choice(['2hand', 'like new', 'cũ', 'mới'])}",
                f"tìm {short_keyword.lower()}",
                f"mua {short_keyword.lower()}"
            ]
            
            generated_rows.append({
                "input": clean_text(random.choice(keyword_variants)),
                "product_title": title,
                "product_description": desc,
                "is_new": is_new
            })
            
    # 3. Tạo DataFrame từ dữ liệu sinh thêm
    df_synthetic = pd.DataFrame(generated_rows)
    
    # 4. Gộp dữ liệu gốc và dữ liệu sinh thêm
    df_combined = pd.concat([df_orig, df_synthetic], ignore_index=True)
    
    # Chuẩn hóa tiền xử lý trước khi lưu
    df_combined['input'] = df_combined['input'].apply(clean_text)
    df_combined['product_title'] = df_combined['product_title'].apply(clean_text)
    
    # Lọc bỏ các dòng bị rỗng
    df_combined = df_combined.dropna(subset=['input', 'product_title'])
    df_combined = df_combined[(df_combined['input'] != "") & (df_combined['product_title'] != "")]
    
    # Xáo trộn ngẫu nhiên dữ liệu để chia Train/Val đều hơn
    df_combined = df_combined.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # 5. Lưu kết quả
    output_path = "data/augmented_dataset.csv"
    df_combined.to_csv(output_path, index=False, encoding="utf-8")
    
    print(f"✅ Hoàn tất sinh dữ liệu! Đã lưu {len(df_combined)} dòng vào {output_path} (Tăng thêm {len(df_synthetic)} dòng).")

if __name__ == "__main__":
    generate_queries()
