from pathlib import Path
import polars as pl
import mlflow
import joblib
from sklearn.linear_model import LogisticRegression

def train():
    repo_root = Path(__file__).resolve().parents[2]
    train_path = repo_root / "data" / "processed" / "train.parquet"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Missing training data: {train_path}. Run the split step first."
        )

    mlflow.start_run()
    # mlflow.autolog()
    df = pl.read_parquet(str(train_path))

    X = df.select(["userId", "movieId"]).to_pandas()
    y = df["liked"].to_pandas()

    model = LogisticRegression()
    model.fit(X, y)

    joblib.dump(model, "model.pkl")
    mlflow.log_param("model", "logistic_regression")
    mlflow.log_artifact("model.pkl")
    mlflow.log_metric("accuracy", model.score(X, y))
    mlflow.end_run()
    print("Training done!!")

if __name__ == "__main__":
    train()