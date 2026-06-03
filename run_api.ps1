$env:DATABASE_URL = $env:DATABASE_URL
if (-not $env:DATABASE_URL) {
  $env:DATABASE_URL = "mysql://temo:Temo@2026@10.94.104.24:9991/temo?charset=utf8mb4"
}

$env:LOAD_VIT5_ON_STARTUP = "true"
$env:SENTENCE_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
