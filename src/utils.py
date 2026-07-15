import os
import pickle
import yaml
from typing import Any


def create_directories(paths: list):
    for path in paths:
        os.makedirs(path, exist_ok=True)


def save_object(file_path: str, obj: Any):
    directory = os.path.dirname(file_path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(file_path, "wb") as file:
        pickle.dump(obj, file)


def load_object(file_path: str):
    with open(file_path, "rb") as file:
        return pickle.load(file)


def read_yaml(file_path: str):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)
    
from sklearn.metrics import accuracy_score


def evaluate_models(X_train, y_train, X_test, y_test, models):
    """
    Train multiple models and return their test accuracies.
    """

    report = {}

    for model_name, model in models.items():

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        report[model_name] = accuracy

    return report