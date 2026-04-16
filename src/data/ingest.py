import polars as pl
import os
import zipfile
import urllib.request

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"

def download_data():
    os.makedirs("data/raw", exist_ok=True)

    zip_path = "data/raw/ml-25m.zip"

    if not os.path.exists(zip_path):
        print("Downloading dataset..")
        urllib.request.urlretrieve(DATA_URL, zip_path)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("data/raw")

    print("Download and Extraction completed")


def load_data():
    ratings = pl.read_csv("data/raw/ml-25m/ratings.csv")
    movies = pl.read_csv("data/raw/ml-25m/movies.csv")
    tags = pl.read_csv("data/raw/ml-25m/tags.csv")

    print("Ratings shape:", ratings.shape)
    print("Movies shape:", movies.shape)
    print("Tags shape:", tags.shape)

    return ratings, movies, tags


if __name__ == "__main__":
    download_data()
    load_data()