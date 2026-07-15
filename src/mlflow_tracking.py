import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature


class MLflowTracker:

    def __init__(self):
        mlflow.set_experiment("Customer Churn Prediction")

    def log_model(
        self,
        model_name,
        model,
        metrics,
        params,
        X_train
    ):

        with mlflow.start_run(run_name=model_name):

            # Log parameters
            mlflow.log_params(params)

            # Log metrics
            mlflow.log_metrics(metrics)

            # Tag
            mlflow.set_tag("model_name", model_name)

            # Infer model signature
            signature = infer_signature(
                X_train,
                model.predict(X_train)
            )

            # Log and Register model
            mlflow.sklearn.log_model(
                sk_model=model,
                name="CustomerChurnModel",
                signature=signature,
                input_example=X_train[:5],
                registered_model_name="CustomerChurnModel"
            )