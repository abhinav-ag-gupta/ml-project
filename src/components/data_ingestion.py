import os
import sys
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging

PROJECT_ROOT = Path(__file__).resolve().parents[2]

@dataclass
class DataIngestionConfig:
    train_data_path: str = str(PROJECT_ROOT / "artifacts" / "train.csv")
    test_data_path: str = str(PROJECT_ROOT / "artifacts" / "test.csv")
    raw_data_path: str = str(PROJECT_ROOT / "artifacts" / "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion component")
        try:
            PROJECT_ROOT = Path(__file__).resolve().parents[2]
            DATA_PATH = PROJECT_ROOT / "notebook" / "data" / "stud.csv"

            logging.info(f"Reading dataset from: {DATA_PATH}")
            df = pd.read_csv(DATA_PATH)

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train-test split initiated")

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion completed successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
