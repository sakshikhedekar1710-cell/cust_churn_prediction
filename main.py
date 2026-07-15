from src.data_ingestion import DataIngestion
from src.data_transformation import DataTransformation
from src.train import ModelTrainer
from src.model_tuner import ModelTuner


if __name__ == "__main__":

    print("\n---------- CUSTOMER CHURN PREDICTION ----------\n")

    # -----------------------------------
    # DATA INGESTION
    # -----------------------------------
    ingestion = DataIngestion()

    train_path, test_path = ingestion.initiate_data_ingestion()

    # -----------------------------------
    # DATA TRANSFORMATION
    # -----------------------------------
    transformation = DataTransformation()

    train_arr, test_arr, _ = (
        transformation.initiate_data_transformation(
            train_path,
            test_path
        )
    )

    # -----------------------------------
    # MODEL TRAINING
    # -----------------------------------
    trainer = ModelTrainer()

    best_model, model_report, metrics = (
        trainer.initiate_model_training(
            train_arr,
            test_arr
        )
    )

    # -----------------------------------
    # HYPERPARAMETER TUNING
    # -----------------------------------
    X_train = train_arr[:, :-1]
    y_train = train_arr[:, -1]

    tuner = ModelTuner()

    tuned_model, best_name, best_score = tuner.tune_models(
        X_train,
        y_train
    )

    print("\n---------- Hyperparameter Tuning ----------\n")

    print("Best Tuned Model :", best_name)
    print("Best Cross Validation Accuracy :", round(best_score, 4))

   