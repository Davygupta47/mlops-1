import polars as pl
import joblib
from sklearn.metrics import accuracy_score

def evaluate():
    df = pl.read_parquet("data/processed/test.parquet")
    X = df.select(["userId", "movieId"]).to_pandas()
    y = df["liked"].to_pandas()

    model = joblib.load("model.pkl")
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    print(f"Accuracy: {acc}")

if __name__ == "__main__":
    evaluate()