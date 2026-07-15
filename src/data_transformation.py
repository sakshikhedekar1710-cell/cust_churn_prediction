import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logger
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join(
        "artifacts",
        "models",
        "preprocessor.pkl"
    )


class DataTransformation:

    def __init__(self):
        self.config = DataTransformationConfig()

    def get_data_transformer_object(self):

        try:

            numerical_columns = [
                "SeniorCitizen",
                "tenure",
                "MonthlyCharges",
                "TotalCharges"
            ]

            categorical_columns = [
                "gender",
                "Partner",
                "Dependents",
                "PhoneService",
                "MultipleLines",
                "InternetService",
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies",
                "Contract",
                "PaperlessBilling",
                "PaymentMethod"
            ]

            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))
                ]
            )

            logger.info("Numerical Pipeline Created")
            logger.info("Categorical Pipeline Created")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numerical_pipeline, numerical_columns),
                    ("cat", categorical_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(
        self,
        train_path,
        test_path
    ):

        try:

            logger.info("Reading Train and Test Data")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # FIX FOR TOTALCHARGES
            train_df["TotalCharges"] = (
                train_df["TotalCharges"]
                .replace(" ", np.nan)
                .pipe(pd.to_numeric, errors="coerce")
            )

            test_df["TotalCharges"] = (
                test_df["TotalCharges"]
                .replace(" ", np.nan)
                .pipe(pd.to_numeric, errors="coerce")
            )

            logger.info("Train and Test Data Loaded")

            preprocessing_obj = self.get_data_transformer_object()

            target_column = "Churn"

            drop_columns = [
                target_column,
                "customerID"
            ]

            X_train = train_df.drop(columns=drop_columns, axis=1)
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=drop_columns, axis=1)
            y_test = test_df[target_column]

            logger.info("Applying Preprocessing")

            X_train = preprocessing_obj.fit_transform(X_train)
            X_test = preprocessing_obj.transform(X_test)

            y_train = np.where(y_train == "Yes", 1, 0)
            y_test = np.where(y_test == "Yes", 1, 0)

            train_arr = np.c_[X_train, y_train]
            test_arr = np.c_[X_test, y_test]

            save_object(
                file_path=self.config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logger.info("Preprocessor Saved Successfully")

            return (
                train_arr,
                test_arr,
                self.config.preprocessor_obj_file_path
            )

        except Exception as e:
            logger.exception("Error in Data Transformation")
            raise CustomException(e, sys)