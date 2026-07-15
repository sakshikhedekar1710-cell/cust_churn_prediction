import json
import os
import sys

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from src.exception import CustomException
from src.logger import logger


class ModelEvaluation:

    def evaluate(
        self,
        model,
        X_test,
        y_test
    ):

        try:

            logger.info("Starting Model Evaluation")

            predictions = model.predict(X_test)

            accuracy = accuracy_score(
                y_test,
                predictions
            )

            precision = precision_score(
                y_test,
                predictions
            )

            recall = recall_score(
                y_test,
                predictions
            )

            f1 = f1_score(
                y_test,
                predictions
            )

            roc = roc_auc_score(
                y_test,
                predictions
            )

            report = classification_report(
                y_test,
                predictions
            )

            print("\nClassification Report\n")
            print(report)

            os.makedirs(
                "artifacts/reports",
                exist_ok=True
            )

            metrics = {

                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "roc_auc": roc
            }

            with open(
                "artifacts/reports/metrics.json",
                "w"
            ) as file:

                json.dump(
                    metrics,
                    file,
                    indent=4
                )

            cm = confusion_matrix(
                y_test,
                predictions
            )

            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm
            )

            disp.plot()

            plt.savefig(
                "artifacts/reports/confusion_matrix.png"
            )

            plt.close()

            with open(
                "artifacts/reports/classification_report.txt",
                "w"
            ) as file:

                file.write(report)

            logger.info("Model Evaluation Completed")

            return metrics

        except Exception as e:
            raise CustomException(e, sys)