import polars as pl
import os

def preprocess():
    df = pl.read_csv("data/raw/data.csv")
    df["rating"] = df["rating"].apply(lambda x:1 if x >= 4 else 0)
    os.makedir("data/processed", exist_ok=True)
    df.write_csv("data/processed/train.csv", index=False)
    print("Preprocessing done!!")

if __name__ == "__main__":    preprocess()