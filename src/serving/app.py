from pathlib import Path
from fastapi import FastAPI, HTTPException
import joblib
import numpy as np

app = FastAPI()
model = None

@app.on_event("startup")
def load_model() -> None:
    global model
    repo_root = Path(__file__).resolve().parents[2]
    model_path = repo_root / "model.pkl"
    model = joblib.load(model_path)


@app.get("/")
def home():
    return {"message": "Movie Recommendation API"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/predict")
def predict(user: int, item: int):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if user <= 0 or item <= 0:
        raise HTTPException(
            status_code=422,
            detail="user and item must be positive integers",
        )
    pred = model.predict(np.array([[user, item]]))
    return {"recommendation": int(pred[0])}
