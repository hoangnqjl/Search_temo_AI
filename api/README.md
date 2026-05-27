# MaroMart ModelAI API

API Python độc lập để test ModelAI bằng Postman trước khi nối vào backend TypeScript.

## Run

```powershell
cd E:\VKU\Projects\MaroMart\ModelAI\Search_temo_AI
.\run_api.ps1
```

Hoặc:

```powershell
$env:DATABASE_URL="mysql://temo:Temo@2026@10.94.104.24:9991/temo?charset=utf8mb4"
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Base URL:

```text
http://localhost:8000
```

## Endpoints

### Health

```http
GET /health
```

### Reload products from MySQL

```http
POST /reload-products
```

### Load local ViT5 model

Server khởi động nhanh bằng DB search trước. Khi muốn dùng model trong `file_train/best_model`, gọi:

```http
POST /load-model
```

### Sample products

```http
GET /products/sample?limit=5
```

### Search

```http
POST /search
Content-Type: application/json
```

Body:

```json
{
  "query": "toi can iphone cu duoi 10 trieu",
  "limit": 5,
  "useModel": false
}
```

### Recommend

```http
POST /recommend
Content-Type: application/json
```

Body:

```json
{
  "query": "toi can dien thoai pin tot de chup anh",
  "limit": 5,
  "userContext": "uu tien gia re, hang like new",
  "useModel": false
}
```

## Notes

- API chỉ lấy sản phẩm `status = 'active'`.
- Product được cache trong RAM, mặc định 5 phút.
- `POST /reload-products` dùng để ép load lại data từ MySQL.
- ViT5 được load từ `file_train/best_model` khi gọi `/load-model` hoặc khi request có `"useModel": true`.
