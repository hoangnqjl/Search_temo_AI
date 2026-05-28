import json
import os
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs

import pymysql
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE_URL = "mysql://temo:Temo@2026@10.94.104.24:9991/temo?charset=utf8mb4"
DEFAULT_MODEL_DIR = ROOT_DIR / "file_train" / "best_model"

PRODUCT_CACHE_TTL_SECONDS = int(os.getenv("PRODUCT_CACHE_TTL_SECONDS", "300"))
DEFAULT_LIMIT = 10
MAX_LIMIT = 50
MAX_DB_PRODUCTS = int(os.getenv("MAX_DB_PRODUCTS", "1000"))

app = FastAPI(title="MaroMart ModelAI API", version="1.0.0")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    limit: int = Field(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT)
    categoryId: Optional[str] = None
    minPrice: Optional[float] = None
    maxPrice: Optional[float] = None
    refresh: bool = False
    useModel: bool = False


class RecommendRequest(SearchRequest):
    userContext: Optional[str] = None


@dataclass
class RuntimeState:
    products: List[Dict[str, Any]]
    loaded_at: float
    db_error: Optional[str] = None
    model_error: Optional[str] = None
    vit5_tokenizer: Any = None
    vit5_model: Any = None
    sentence_model: Any = None
    torch: Any = None


state = RuntimeState(products=[], loaded_at=0)


def log(message: str) -> None:
    print(f"[ModelAI] {message}", flush=True)


def parse_mysql_url(url: str) -> Dict[str, Any]:
    if not url.startswith("mysql://"):
        raise ValueError("DATABASE_URL must start with mysql://")

    raw = url[len("mysql://") :]
    authority_and_path, _, query_string = raw.partition("?")
    authority, _, database = authority_and_path.rpartition("/")
    user_password, _, host_port = authority.rpartition("@")

    if not user_password or not host_port or not database:
        raise ValueError("Invalid DATABASE_URL format")

    user, _, password = user_password.partition(":")
    host, sep, port = host_port.partition(":")
    query = parse_qs(query_string)

    return {
        "host": host,
        "port": int(port) if sep and port else 3306,
        "user": user,
        "password": password,
        "database": database,
        "charset": query.get("charset", ["utf8mb4"])[0],
        "cursorclass": pymysql.cursors.DictCursor,
        "connect_timeout": 8,
        "read_timeout": 20,
        "write_timeout": 20,
    }


def get_database_url() -> str:
    return os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)


def get_connection():
    return pymysql.connect(**parse_mysql_url(get_database_url()))


def safe_json(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return default
    return default


def fetch_products_from_db(limit: int = MAX_DB_PRODUCTS) -> List[Dict[str, Any]]:
    sql = """
        SELECT
            p.productId,
            p.categoryId,
            p.userId,
            p.productName,
            p.productPrice,
            p.productDescription,
            p.productCondition,
            p.productBrand,
            p.productWP,
            p.productOrigin,
            p.productCategory,
            p.productAttribute,
            p.productAddress,
            p.productMedia,
            p.marketPrice,
            p.status,
            p.createdAt,
            p.updatedAt,
            c.categoryName
        FROM products p
        LEFT JOIN categories c ON c.categoryId = p.categoryId
        WHERE p.status = 'active'
        ORDER BY p.createdAt DESC
        LIMIT %s
    """

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, (limit,))
            rows = cursor.fetchall()

    products: List[Dict[str, Any]] = []
    for row in rows:
        row["productAttribute"] = safe_json(row.get("productAttribute"), {})
        row["productAddress"] = safe_json(row.get("productAddress"), {})
        row["productMedia"] = safe_json(row.get("productMedia"), [])
        for key in ("createdAt", "updatedAt"):
            if row.get(key) is not None:
                row[key] = row[key].isoformat()
        products.append(row)

    return products


def load_products(force: bool = False) -> List[Dict[str, Any]]:
    now = time.time()
    cache_expired = now - state.loaded_at > PRODUCT_CACHE_TTL_SECONDS
    if state.products and not force and not cache_expired:
        return state.products

    try:
        state.products = fetch_products_from_db()
        state.loaded_at = now
        state.db_error = None
        log(f"Loaded {len(state.products)} active products from MySQL.")
    except Exception as exc:
        state.db_error = str(exc)
        log(f"Database load failed: {exc}")
        if not state.products:
            raise

    return state.products


def strip_accents(value: str) -> str:
    if not value:
        return ""
    value = value.replace("đ", "d").replace("Đ", "d")
    normalized = unicodedata.normalize("NFD", value)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def normalize_text(value: str) -> str:
    value = strip_accents(value).lower()
    value = re.sub(r"[^a-z0-9\s]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def flatten_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        parts: List[str] = []
        for key, item in value.items():
            parts.append(str(key))
            parts.append(flatten_value(item))
        return " ".join(parts)
    if isinstance(value, list):
        return " ".join(flatten_value(item) for item in value)
    return str(value)


def product_text(product: Dict[str, Any]) -> str:
    fields = [
        product.get("categoryName"),
        product.get("productCategory"),
        product.get("productName"),
        product.get("productBrand"),
        product.get("productCondition"),
        product.get("productDescription"),
        flatten_value(product.get("productAttribute")),
        flatten_value(product.get("productAddress")),
    ]
    return " ".join(str(field) for field in fields if field)


STOPWORDS = {
    "toi",
    "can",
    "tim",
    "kiem",
    "muon",
    "mua",
    "mot",
    "cai",
    "chiec",
    "san",
    "pham",
    "hang",
    "gia",
    "duoi",
    "tren",
    "tam",
    "khoang",
    "cho",
    "minh",
    "ban",
    "co",
    "khong",
    "va",
    "de",
    "re",
    "tot",
}


def query_tokens(query: str) -> List[str]:
    return [
        token
        for token in normalize_text(query).split()
        if len(token) >= 2 and token not in STOPWORDS
    ]


def extract_budget(query: str) -> Optional[float]:
    q = normalize_text(query).replace(",", ".")
    patterns = [
        (r"(\d+(?:\.\d+)?)\s*(?:trieu|tr|m)\b", 1_000_000),
        (r"(\d+(?:\.\d+)?)\s*(?:ngan|k)\b", 1_000),
    ]
    for pattern, multiplier in patterns:
        match = re.search(pattern, q)
        if match:
            return float(match.group(1)) * multiplier

    direct = re.search(r"\b(\d{1,3}(?:[.]\d{3})+|\d{6,})\s*(?:vnd|dong|d)?\b", q)
    if direct:
        return float(direct.group(1).replace(".", ""))
    return None


def filter_products(
    products: List[Dict[str, Any]],
    category_id: Optional[str],
    min_price: Optional[float],
    max_price: Optional[float],
) -> List[Dict[str, Any]]:
    result = products
    if category_id:
        normalized_category = normalize_text(category_id)
        result = [
            product
            for product in result
            if normalize_text(str(product.get("categoryId", ""))) == normalized_category
            or normalize_text(str(product.get("categoryName", ""))) == normalized_category
        ]
    if min_price is not None:
        result = [product for product in result if float(product.get("productPrice") or 0) >= min_price]
    if max_price is not None:
        result = [product for product in result if float(product.get("productPrice") or 0) <= max_price]
    return result


def load_vit5_model() -> None:
    if state.vit5_model is not None and state.vit5_tokenizer is not None:
        return

    model_dir = Path(os.getenv("VIT5_MODEL_DIR", str(DEFAULT_MODEL_DIR)))
    if not model_dir.exists():
        state.model_error = f"Model folder not found: {model_dir}"
        return

    try:
        log(f"Loading ViT5 model from {model_dir}")
        import torch
        from transformers import T5ForConditionalGeneration, T5Tokenizer

        state.torch = torch
        state.vit5_tokenizer = T5Tokenizer.from_pretrained(str(model_dir), legacy=False)
        state.vit5_model = T5ForConditionalGeneration.from_pretrained(str(model_dir))
        state.vit5_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        state.vit5_model.eval()
        state.model_error = None
        log(f"Loaded ViT5 model from {model_dir}")
    except Exception as exc:
        state.model_error = str(exc)
        log(f"ViT5 model unavailable: {exc}")


def load_sentence_model() -> None:
    model_name = os.getenv("SENTENCE_MODEL_NAME", "")
    if not model_name:
        return

    try:
        from sentence_transformers import SentenceTransformer

        state.sentence_model = SentenceTransformer(model_name)
        log(f"Loaded sentence model: {model_name}")
    except Exception as exc:
        log(f"Sentence model unavailable: {exc}")


def refine_query(query: str) -> str:
    if not state.vit5_model or not state.vit5_tokenizer or not state.torch:
        return query

    try:
        torch = state.torch
        text = "intent: " + query
        inputs = state.vit5_tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(state.vit5_model.device)
        with torch.no_grad():
            outputs = state.vit5_model.generate(
                **inputs,
                max_length=64,
                num_beams=5,
                repetition_penalty=2.5,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )
        refined = state.vit5_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return refined or query
    except Exception as exc:
        log(f"Query refinement failed: {exc}")
        return query


def lexical_score(query: str, refined_query: str, product: Dict[str, Any], budget: Optional[float]) -> Tuple[float, str]:
    text = normalize_text(product_text(product))
    combined_query = f"{query} {refined_query}".strip()
    tokens = query_tokens(combined_query)
    if not tokens:
        return 0.0, "Không đủ từ khóa để so khớp"

    matched_tokens = [token for token in tokens if token in text]
    score = len(set(matched_tokens)) / max(1, len(set(tokens)))

    product_name = normalize_text(str(product.get("productName", "")))
    brand = normalize_text(str(product.get("productBrand", "")))
    normalized_query = normalize_text(combined_query)

    if product_name and any(token in product_name for token in tokens):
        score += 0.15
    if brand and brand in normalized_query:
        score += 0.12

    price = float(product.get("productPrice") or 0)
    if budget:
        if price <= budget:
            score += 0.12
        elif price <= budget * 1.2:
            score *= 0.9
        else:
            score *= max(0.1, min(1.0, budget / max(price, 1)))

    reason = "Khớp từ khóa: " + (", ".join(sorted(set(matched_tokens))) if matched_tokens else "ít")
    if budget:
        reason += f"; ngân sách nhận diện khoảng {int(budget):,} VND"
    return score, reason


def sentence_scores(query: str, products: List[Dict[str, Any]]) -> Optional[List[float]]:
    if not state.sentence_model:
        return None
    try:
        import numpy as np

        texts = [product_text(product) for product in products]
        query_vec = state.sentence_model.encode([query], normalize_embeddings=True)[0]
        product_vecs = state.sentence_model.encode(texts, normalize_embeddings=True)
        return (product_vecs @ query_vec).astype(float).tolist()
    except Exception as exc:
        log(f"Sentence scoring failed: {exc}")
        return None


def rank_products(
    query: str,
    products: List[Dict[str, Any]],
    limit: int,
    use_model: bool = False,
) -> Dict[str, Any]:
    if use_model:
        load_vit5_model()

    refined_query = refine_query(query) if use_model else query
    budget = extract_budget(query)
    semantic_scores = sentence_scores(refined_query, products)

    ranked: List[Dict[str, Any]] = []
    for index, product in enumerate(products):
        lexical, reason = lexical_score(query, refined_query, product, budget)
        semantic = semantic_scores[index] if semantic_scores is not None else None
        score = (semantic * 0.7 + lexical * 0.3) if semantic is not None else lexical

        item = dict(product)
        item["score"] = round(float(score), 6)
        item["reason"] = reason
        if semantic is not None:
            item["semanticScore"] = round(float(semantic), 6)
        ranked.append(item)

    ranked.sort(key=lambda item: item["score"], reverse=True)
    return {
        "refinedQuery": refined_query,
        "budget": budget,
        "items": ranked[:limit],
    }


def success_response(query: str, ranked: Dict[str, Any], source: str) -> Dict[str, Any]:
    return {
        "success": True,
        "query": query,
        "refinedQuery": ranked["refinedQuery"],
        "budget": ranked["budget"],
        "count": len(ranked["items"]),
        "source": source,
        "model": {
            "vit5Loaded": state.vit5_model is not None,
            "sentenceModelLoaded": state.sentence_model is not None,
            "modelError": state.model_error,
        },
        "data": ranked["items"],
    }


from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.getenv("LOAD_VIT5_ON_STARTUP", "false").lower() == "true":
        load_vit5_model()
    load_sentence_model()
    try:
        load_products(force=True)
    except Exception:
        pass
    yield

app.router.lifespan_context = lifespan


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "success": True,
        "dbConnected": state.db_error is None and bool(state.products),
        "productCount": len(state.products),
        "lastLoadedAt": state.loaded_at,
        "dbError": state.db_error,
        "vit5Loaded": state.vit5_model is not None,
        "sentenceModelLoaded": state.sentence_model is not None,
        "modelError": state.model_error,
    }


@app.post("/reload-products")
def reload_products() -> Dict[str, Any]:
    products = load_products(force=True)
    return {"success": True, "count": len(products), "dbError": state.db_error}


@app.post("/load-model")
def load_model() -> Dict[str, Any]:
    load_vit5_model()
    return {
        "success": state.vit5_model is not None,
        "vit5Loaded": state.vit5_model is not None,
        "modelError": state.model_error,
    }


@app.get("/products/sample")
def product_sample(limit: int = Query(5, ge=1, le=50)) -> Dict[str, Any]:
    products = load_products()
    return {
        "success": True,
        "count": min(limit, len(products)),
        "data": products[:limit],
        "dbError": state.db_error,
    }


@app.post("/search")
def search(request: SearchRequest) -> Dict[str, Any]:
    products = load_products(force=request.refresh)
    filtered = filter_products(products, request.categoryId, request.minPrice, request.maxPrice)
    ranked = rank_products(request.query, filtered, request.limit, request.useModel)
    source = "mysql+vit5+lexical" if request.useModel else "mysql+lexical"
    return success_response(request.query, ranked, source)


@app.get("/search")
def search_get(
    q: str = Query(..., min_length=1),
    limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    categoryId: Optional[str] = None,
    minPrice: Optional[float] = None,
    maxPrice: Optional[float] = None,
    useModel: bool = False,
) -> Dict[str, Any]:
    request = SearchRequest(
        query=q,
        limit=limit,
        categoryId=categoryId,
        minPrice=minPrice,
        maxPrice=maxPrice,
        useModel=useModel,
    )
    return search(request)


@app.post("/recommend")
def recommend(request: RecommendRequest) -> Dict[str, Any]:
    query = request.query
    if request.userContext:
        query = f"{query}. Ngữ cảnh người dùng: {request.userContext}"
    products = load_products(force=request.refresh)
    filtered = filter_products(products, request.categoryId, request.minPrice, request.maxPrice)
    ranked = rank_products(query, filtered, request.limit, request.useModel)
    source = "mysql+vit5+advisor" if request.useModel else "mysql+advisor"
    response = success_response(request.query, ranked, source)
    response["advice"] = build_advice(request.query, ranked["items"])
    return response


def build_advice(query: str, items: List[Dict[str, Any]]) -> str:
    if not items:
        return "Chưa tìm thấy sản phẩm phù hợp với nhu cầu này."
    best = items[0]
    name = best.get("productName", "sản phẩm này")
    price = best.get("productPrice", 0)
    condition = best.get("productCondition", "")
    return (
        f"Gợi ý ưu tiên là {name} giá khoảng {int(float(price or 0)):,} VND"
        f"{' tình trạng ' + condition if condition else ''}. "
        f"Sản phẩm này có điểm phù hợp cao nhất với nhu cầu: {query}."
    )


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("AI_HOST", "0.0.0.0")
    port = int(os.getenv("AI_PORT", "8000"))

    # Add parent directory of 'api' to sys.path to allow importing 'api.main'
    parent_dir = str(Path(__file__).resolve().parents[1])
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    uvicorn.run("api.main:app", host=host, port=port, reload=False)
