from src.data.ingest import download_data, load_data
from src.data.preprocess import preprocess
from src.features.build import split_data
from src.models.train import train
from src.models.eval import evaluate

def run_pipeline():
    print("Starting pipeline...")

    download_data()
    load_data()

    preprocess()
    split_data()

    train()
    evaluate()

    print("Pipeline completed!")

if __name__ == "__main__":
    run_pipeline()