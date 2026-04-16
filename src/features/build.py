import polars as pl
import yaml
from sklearn.model_selection import train_test_split

def split_data():
    config = yaml.safe_load(open("src/config/config.yaml"))
    df = pl.read_parquet("data/processed/data.parquet")
    pdf = df.to_pandas()
    
    train, test = train_test_split(
        pdf,test_size=config["model"]["test_size"],random_state=config["model"]["random_state"]
    )

    pl.from_pandas(train).write_parquet("data/processed/train.parquet")
    pl.from_pandas(test).write_parquet("data/processed/test.parquet")

    print("Data split done!!")
if __name__ == "__main__":    split_data()