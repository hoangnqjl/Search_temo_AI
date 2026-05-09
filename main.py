import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from api.search_service import APISearchService
from build.search_service import LocalSearchService
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="ModelAI - Semantic Search Service")

# Initialize models
try:
    api_service = APISearchService()
    print("API Search Service initialized.")
except Exception as e:
    api_service = None
    print(f"API Search Service could not be initialized: {e}")

# Note: Local model might take a few moments to download on first run
local_service = LocalSearchService()
print("Local Search Service (Transformer) initialized.")

class EmbeddingRequest(BaseModel):
    text: str
    mode: Optional[str] = "local" # or "api"

class BatchEmbeddingRequest(BaseModel):
    texts: List[str]
    mode: Optional[str] = "local"

class TestEmbedRequest(BaseModel):
    query: str
    candidates: List[str]
    mode: Optional[str] = "local"

@app.post("/embed")
async def get_embedding(req: EmbeddingRequest):
    """Generate an embedding for a piece of text."""
    try:
        if req.mode == "api":
            if not api_service:
                raise HTTPException(status_code=500, detail="API Service not properly configured")
            return {"embedding": api_service.get_embedding(req.text)}
        else:
            return {"embedding": local_service.get_embedding(req.text)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed/batch")
async def get_embeddings_batch(req: BatchEmbeddingRequest):
    """Generate embeddings for a batch of texts."""
    try:
        if req.mode == "api":
            if not api_service:
                raise HTTPException(status_code=500, detail="API Service not properly configured")
            return {"embeddings": api_service.get_embeddings_batch(req.texts)}
        else:
            return {"embeddings": local_service.get_embeddings_batch(req.texts)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/test-embed")
async def test_semantic_matching(req: TestEmbedRequest):
    """Thử nghiệm so sánh ngữ nghĩa trực tiếp trên Python."""
    try:
        # 1. Tạo vector cho câu query
        if req.mode == "api":
            if not api_service: raise HTTPException(status_code=500, detail="API Service not configured")
            q_vec = api_service.get_embedding(req.query)
            c_vecs = api_service.get_embeddings_batch(req.candidates)
        else:
            q_vec = local_service.get_embedding(req.query)
            c_vecs = local_service.get_embeddings_batch(req.candidates)
        
        results = []
        for i, cand_text in enumerate(req.candidates):
            # 2. Tính độ tương đồng
            score = local_service.compute_similarity(q_vec, c_vecs[i])
            results.append({"text": cand_text, "score": float(score)})
        
        # 3. Sắp xếp kết quả (Điểm cao nhất lên đầu)
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return {
            "query": req.query,
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "paraphrase-multilingual-MiniLM-L12-v2"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
