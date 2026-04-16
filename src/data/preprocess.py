import polars as pl
import os

def preprocess():
    ratings = pl.read_csv("data/raw/ml-25m/ratings.csv")
    movies = pl.read_csv("data/raw/ml-25m/movies.csv")

    ratings = ratings.with_columns(
        pl.from_epoch("timestamp").alias("datetime")
    )
#Implicit feedback (like/dislike)
    ratings = ratings.with_columns(
        (pl.col("rating") >= 3.5).cast(pl.Int8).alias("liked")
    )
#Filtering active users
    user_counts = ratings.group_by("userId").count()
    active_users = user_counts.filter(pl.col("count") > 50)["userId"]

    ratings = ratings.filter(pl.col("userId").is_in(active_users))
    df = ratings.join(movies, on="movieId")
#Extraction of genres into list
    df = df.with_columns(
        pl.col("genres").str.split("|")
    )

    os.makedirs("data/processed", exist_ok=True)
    df.write_parquet("data/processed/data.parquet")

    print("Preprocessing complete")
    print(df.head())


if __name__ == "__main__":
    preprocess()