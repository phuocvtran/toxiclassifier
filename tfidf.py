from preprocess import preprocess_text_list
import joblib


class Tfidf:
    def __init__(self, vectorizer_path):
        self.vectorizer = joblib.load(vectorizer_path)

    def transform(self, text_list):
        preprocessed_text_list = preprocess_text_list(text_list)
        return self.vectorizer.transform(preprocessed_text_list)
