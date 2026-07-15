import os
import sys

from dataclasses import dataclass

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.logger import logger
from src.exception import CustomException
from src.utils import save_object


@dataclass
class ModelTunerConfig:
    tuned_model_path = os.path.join(
        "artifacts",
        "models",
        "best_tuned_model.pkl"
    )


class ModelTuner:

    def __init__(self):
        self.config = ModelTunerConfig()

    def tune_models(
        self,
        X_train,
        y_train
    ):

        try:

            logger.info("Starting Hyperparameter Tuning")

            models = {

                "Random Forest": (
                    RandomForestClassifier(random_state=42),
                    {
                        "n_estimators": [100, 200],
                        "max_depth": [10, 20],
                        "min_samples_split": [2, 5]
                    }
                ),

                "XGBoost": (
                    XGBClassifier(
                        eval_metric="logloss",
                        random_state=42
                    ),
                    {
                        "n_estimators": [100, 200],
                        "learning_rate": [0.05, 0.1],
                        "max_depth": [3, 5]
                    }
                )
            }

            best_score = 0
            best_model = None
            best_name = ""

            for name, (model, params) in models.items():

                grid = GridSearchCV(
                    estimator=model,
                    param_grid=params,
                    cv=5,
                    scoring="accuracy",
                    n_jobs=-1
                )

                grid.fit(
                    X_train,
                    y_train
                )

                print(
                    f"{name} Best Score : {grid.best_score_:.4f}"
                )

                print(
                    f"{name} Best Params : {grid.best_params_}"
                )

                if grid.best_score_ > best_score:

                    best_score = grid.best_score_

                    best_model = grid.best_estimator_

                    best_name = name

            save_object(
                self.config.tuned_model_path,
                best_model
            )

            logger.info("Hyperparameter Tuning Completed")

            return (
                best_model,
                best_name,
                best_score
            )

        except Exception as e:

            raise CustomException(e, sys)