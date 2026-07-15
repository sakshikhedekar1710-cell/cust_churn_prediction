import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logger


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")


class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):

        logger.info("Entered Data Ingestion Component")

        try:

            dataset_path = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"

            if not os.path.exists(dataset_path):
                raise FileNotFoundError(
                    f"Dataset not found at {dataset_path}"
                )

            df = pd.read_csv(dataset_path)

            logger.info("Dataset Loaded Successfully")
            logger.info(f"Dataset Shape : {df.shape}")

            os.makedirs("artifacts", exist_ok=True)

            df.to_csv(
                self.ingestion_config.raw_data_path,
                index=False,
                header=True
            )

            train_set, test_set = train_test_split(
                df,
                test_size=0.20,
                random_state=42,
                shuffle=True
            )

            train_set.to_csv(
                self.ingestion_config.train_data_path,
                index=False,
                header=True
            )

            test_set.to_csv(
                self.ingestion_config.test_data_path,
                index=False,
                header=True
            )

            logger.info("Train and Test datasets saved")
            logger.info("Data Ingestion Completed Successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logger.exception("Exception occurred during Data Ingestion")
            raise CustomException(e, sys)