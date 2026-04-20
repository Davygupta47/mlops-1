from pathlib import Path
import polars as pl
import joblib
from sklearn.metrics import accuracy_score

def evaluate():
    repo_root = Path(__file__).resolve().parents[2]
    test_path = repo_root / "data" / "processed" / "test.parquet"
    model_path = repo_root / "model.pkl"

    if not test_path.exists():
        raise FileNotFoundError(
            f"Missing test data: {test_path}. Run the split step first."
        )
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing trained model: {model_path}. Run the training step first."
        )

    df = pl.read_parquet(str(test_path))
    df = df.filter(
        pl.col("liked").is_not_null()
        & pl.col("userId").is_not_null()
        & pl.col("movieId").is_not_null()
    )
    df = df.with_columns(pl.col("liked").cast(pl.Int64))
    X = df.select(["userId", "movieId"]).to_pandas()
    y = df["liked"].to_pandas()

    model = joblib.load("model.pkl")
    #model = joblib.load(model_path)
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    print(f"Accuracy: {acc}")

if __name__ == "__main__":
    evaluate()