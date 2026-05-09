from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import os

def fine_tune_model(training_data=None, model_save_path='model/my-custom-model'):
    # 1. Load the pre-trained model (paraphrase-multilingual-MiniLM-L12-v2 is the base)
    model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
    model = SentenceTransformer(model_name)

    # 2. Prepare training data
    # training_data should be a list of pairs (query, product_description)
    if training_data is None:
        # These are examples of "Semantic Matching"
        # We teach the model that "ai phôn" is close to "iPhone 15 Pro Max"
        training_data = [
            InputExample(texts=['tôi muốn mua ai phôn', 'iPhone 15 Pro Max 256GB Gold']),
            InputExample(texts=['cần mua máy giặt cũ giá rẻ', 'Máy giặt Panasonic 7kg Inverter mới 90%']),
            InputExample(texts=['tìm áo khoác nam size L', 'Áo khoác phao lông vũ Uniqlo cho nam']),
            InputExample(texts=['tủ lạnh tiết kiệm điện', 'Tủ lạnh Samsung 200L Inverter thế hệ mới']),
            InputExample(texts=['xe máy chạy ít km', 'Xe Honda AirBlade 125cc đi được 5000km']),
        ]

    # 3. Configure DataLoader
    # Batch size can be adjusted based on GPU/CPU memory
    train_dataloader = DataLoader(training_data, shuffle=True, batch_size=16)
    
    # Using MultipleNegativesRankingLoss is the industry standard for search/retrieval
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # 4. Starting the Fine-tuning process
    print(f"Starting fine-tuning for {model_name}...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1, # Start with 1 epoch and analyze results
        warmup_steps=100
    )

    # 5. Save the trained model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    print(f"Success! Your custom AI model is saved at: {model_save_path}")

if __name__ == "__main__":
    fine_tune_model()
