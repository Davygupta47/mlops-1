import polars as pl
import os

def preprocess():
    print("Reading CSVs...")
    ratings = pl.scan_csv("data/raw/ml-25m/ratings.csv")
    movies = pl.scan_csv("data/raw/ml-25m/movies.csv")

    print("Transforming...")
    ratings = ratings.with_columns(
        pl.from_epoch("timestamp").alias("datetime"),
        (pl.col("rating") >= 3.5).cast(pl.Int8).alias("liked")
    )

    # Filter active users (>50 ratings) using semi-join — no Python list conversion
    active_users = (
        ratings.group_by("userId")
        .count()
        .filter(pl.col("count") > 50)
        .select("userId")
    )
    ratings = ratings.join(active_users, on="userId", how="semi")

    # Join with movies and split genres
    df = (
        ratings.join(movies, on="movieId")
        .with_columns(pl.col("genres").str.split("|"))
    )

    print("Collecting results...")
    df = df.collect()

    os.makedirs("data/processed", exist_ok=True)
    df.write_parquet("data/processed/data.parquet")

    print("Preprocessing complete")
    print(df.head())


if __name__ == "__main__":
    preprocess()