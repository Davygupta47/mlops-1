import polars as pl
import os

def load_data():
    url = "https://files.grouplens.org/datasets/movielens/ml-100k/u.data"
    df = pl.read_csv(url, sep="\t", names=["user", "item", "rating", "timestamp"])

    os.makedirs("data", exist_ok=True)
    df.to_csv("data//raw/data.csv", index=False)
    print(f"Data ingested done!!")

if __name__ == "__main__":
    load_data()