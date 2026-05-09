import torch
import sys
import os
from transformers import T5ForConditionalGeneration, T5Tokenizer

def run_demo(query):
    print(f"\n🔍 Đang phân tích câu hỏi: '{query}'")
    
    # Tự động chọn model: Ưu tiên model đã train trên Drive, nếu không có thì dùng model gốc
    IN_COLAB = 'google.colab' in sys.modules
    model_path = "VietAI/vit5-base" # Mặc định
    
    if IN_COLAB:
        # Đường dẫn trên Drive (Bạn có thể sửa lại cho đúng folder lưu model của mình)
        drive_model = "/content/drive/MyDrive/Temo/search/file_train/best_model"
        if os.path.exists(drive_model):
            model_path = drive_model
            print(f"✅ Đã tìm thấy model đã huấn luyện trên Drive: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = T5Tokenizer.from_pretrained("VietAI/vit5-base", legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # Encode input
    inputs = tokenizer.encode_plus(
        query,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    ).to(device)

    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=256,
            num_beams=4,
            early_stopping=True
        )

    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("-" * 50)
    print(f"🤖 AI trả về kết quả:\n{decoded_output}")
    print("-" * 50)

if __name__ == "__main__":
    # Lưu ý: Hiện tại demo đang chạy trên model gốc (base). 
    # Sau khi bạn train xong trên Colab và tải trọng số về, hãy thay model_path bằng đường dẫn tới folder đó.
    test_query = "Tôi muốn tìm một chiếc điện thoại iPhone cũ pin còn tốt tầm 5 triệu"
    run_demo(test_query)
