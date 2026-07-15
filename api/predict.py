import pandas as pd

from src.utils import load_object


class Predictor:

    def __init__(self):

        self.model = load_object(
            "artifacts/models/model.pkl"
        )

        self.preprocessor = load_object(
            "artifacts/models/preprocessor.pkl"
        )

    def predict(self, customer):

        df = pd.DataFrame([customer])

        transformed = self.preprocessor.transform(df)

        prediction = self.model.predict(transformed)

        return int(prediction[0])