import polars as pl
import yaml
from pathlib import Path

def split_data():
    print("Starting data split...", flush=True)
    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "src" / "config" / "config.yaml"
    data_path = repo_root / "data" / "processed" / "data.parquet"
    train_path = repo_root / "data" / "processed" / "train.parquet"
    test_path = repo_root / "data" / "processed" / "test.parquet"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    if not data_path.exists():
        raise FileNotFoundError(
            f"Missing processed data: {data_path}. Run preprocessing first."
        )

    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    test_size = float(config["model"]["test_size"])
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"model.test_size must be between 0 and 1, got {test_size}")
    random_state = int(config["model"]["random_state"])

    print(
        f"Splitting {data_path} with test_size={test_size}",
        flush=True,
    )

    # Stream the split to avoid materializing the full dataset in memory.
    split_mod = 1000
    test_cut = int(test_size * split_mod)
    if test_cut <= 0 or test_cut >= split_mod:
        raise ValueError(
            f"model.test_size must yield 1..{split_mod - 1} buckets, got {test_size}"
        )

    lf = pl.scan_parquet(str(data_path)).with_row_index("row_index")
    test_mask = ((pl.col("row_index") + random_state) % split_mod) < test_cut
    train_lf = lf.filter(~test_mask).drop("row_index")
    test_lf = lf.filter(test_mask).drop("row_index")

    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_lf.sink_parquet(str(train_path))
    test_lf.sink_parquet(str(test_path))

    print(f"Wrote train split to {train_path}", flush=True)
    print(f"Wrote test split to {test_path}", flush=True)
    print("Data split done!!", flush=True)


if __name__ == "__main__":
    split_data()
