# Realistic Room i2i (uv, port 7000) – Stability / Replicate / Vertex

## Quickstart (local)
1) Install uv
```bash
pip install uv
```
2) Install deps & set env
```bash
cd backend
uv pip install -e .
cp .env.example .env
# Fill keys: STABILITY_API_KEY or REPLICATE_API_TOKEN or Vertex vars
```
3) Run
```bash
uv run uvicorn main:app --host 0.0.0.0 --port 7000 --reload
```

## Endpoint
- POST `/api/realistic-room` (multipart/form-data)
  - image: file
  - provider: stability | replicate | vertex
  - prompt (optional) – tuned for cozy bedroom realism
  - strength (default 0.6)
  - guidance (default 7.5)

## Vertex requirements
- gcloud CLI installed (host/container)
- Service account JSON with Vertex AI User permission
- `.env` points `GCP_SERVICE_ACCOUNT_JSON_PATH` to that file
- In Docker: put `backend/service-account.json` and run compose

## Docker
```bash
docker compose --env-file backend/.env up --build -d
```
