import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from src.mlflow_tracking import MLflowTracker
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from xgboost import XGBClassifier

from src.evaluate import ModelEvaluation
from src.exception import CustomException
from src.logger import logger
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join(
        "artifacts",
        "models",
        "model.pkl"
    )


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(
        self,
        train_array,
        test_array
    ):

        try:

            logger.info("Splitting Training and Testing Arrays")

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            models = {

                "Logistic Regression":
                    LogisticRegression(max_iter=1000),

                "Decision Tree":
                    DecisionTreeClassifier(random_state=42),

                "Random Forest":
                    RandomForestClassifier(random_state=42),

                "Gradient Boosting":
                    GradientBoostingClassifier(random_state=42),

                "XGBoost":
                    XGBClassifier(
                        eval_metric="logloss",
                        random_state=42
                    )
            }

            logger.info("Evaluating Models")

            model_report = evaluate_models(
                X_train,
                y_train,
                X_test,
                y_test,
                models
            )

            print("\n========== Model Accuracy Scores ==========\n")

            for model_name, score in model_report.items():
                print(f"{model_name:<25}: {score:.4f}")

            best_model_name = max(
                model_report,
                key=model_report.get
            )

            best_model = models[best_model_name]

            print(f"\n✅ Best Model: {best_model_name}")

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logger.info("Best Model Saved Successfully")

            # -------------------------------
            # Evaluate Best Model
            # -------------------------------

            evaluation = ModelEvaluation()

            metrics = evaluation.evaluate(
                best_model,
                X_test,
                y_test
            )
            tracker = MLflowTracker()

            tracker.log_model(
            model_name=best_model_name,
            model=best_model,
            metrics=metrics,
            params=best_model.get_params(),
            X_train=X_train
        )

            print("\n========== Evaluation Metrics ==========\n")

            for metric, value in metrics.items():
                print(f"{metric:<12}: {value:.4f}")

            logger.info("Model Evaluation Completed Successfully")

            return (
                best_model_name,
                model_report,
                metrics
            )

        except Exception as e:
            logger.exception("Exception occurred during Model Training")
            raise CustomException(e, sys)