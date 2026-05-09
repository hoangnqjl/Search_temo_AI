import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer

def run_demo(query, model_path="VietAI/vit5-base"):
    print(f"\n🔍 Đang tìm kiếm cho câu hỏi: '{query}'")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base")
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
