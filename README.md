## MLOps-1
MovieLens MLOps pipeline for ingesting data, preprocessing, splitting, training, evaluation, and serving a FastAPI model.

## Requirements

- Python
- `uv` (https://github.com/astral-sh/uv)
- Docker 

## Clone

```bash
git clone https://github.com/Davygupta47/mlops-1.git
cd mlops-1
```

## Local Run (uv)

Install dependencies:

```bash
uv sync
```

Run the full DVC pipeline:

```bash
uv run dvc repro
```

Run individual steps:

```bash
uv run python src/data/ingest.py
uv run python src/data/preprocess.py
uv run python src/features/build.py
uv run python src/models/train.py
uv run python src/models/eval.py
```

Serve the API locally:

```bash
uv run python -m uvicorn src.serving.app:app --reload
```

Endpoints:

- `GET /`
- `GET /health`
- `GET /predict?user=<id>&item=<id>`

## Docker

Build the image:

```bash
docker build -t mlops-1 .
```

Run the container:

```bash
docker run -p 8000:8000 mlops-1
```

Then open:

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/health`

## Notes

- Large parquet files are ignored by git. Generate them locally via the pipeline.
- If DVC push is needed, configure a remote first:

```bash
uv run dvc remote add -d <remote-name> <remote-url>
```
