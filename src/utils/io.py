import polars as pl
def read_parquet(path):
    return pl.read_parquet(path)

def write_parquet(df, path):
    df.write_parquet(path)