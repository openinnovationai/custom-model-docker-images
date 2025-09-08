## FastAPI App

### Setup

1. Create and activate a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

### Run (dev)

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 and docs at http://localhost:8000/docs

### Endpoints

- `GET /health` — returns `{ "status": "ok" }`
- `GET /` — welcome message and links
- `POST /edit` — upload an image file (form field `file`), placeholder response

### Docker

```bash
docker build -t fastapi-app .
docker run --rm -p 8000:8000 fastapi-app
```

