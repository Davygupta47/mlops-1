import polars as polars
import mlflow 
import joblib
from sklearn.linear_model import LogisticRegression

def train():
    mlfow.start_run()
#   mlflow.autolog()
    df = pl.read_parquet("data/processed/train_parquet")

    X = df.select(["userId", "movieId"]).to_pandas()
    y = df["liked"].to_pandas()

    model = LogisticRegression()
    model.fit(X, y)

    joblib.dump(model, "model.pkl")
    mlflow.log_param("model","logistic_regression")
    mlflow.log_artifact("model.pkl")
    mlflow.log_metric("accuracy",model.score(X, y))
    mlflow.end_run()
    print("Training done!!")

if __name__ == "__main__":
    train()