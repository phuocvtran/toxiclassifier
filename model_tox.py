import joblib


class ModelTox:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def predict(self, vec):
        return self.model.predict(vec)
