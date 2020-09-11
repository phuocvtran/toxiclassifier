import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score


def get_learning_curve(n_list, cooked_data):
    sub_f1_train = []
    sub_f1_valid = []
    sub_f1_test = []
    sub_lg = LogisticRegression(random_state=41,
                        solver="liblinear",
                        C=1.8,
                        penalty="l1",
                        max_iter=1000)
    sub_vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=7000)
    for n in n_list:
        sub_data = cooked_data.sample(n=n, random_state=41)
        sub_X = sub_data.drop(["is_toxic", "id"], axis=1)
        sub_y = sub_data.is_toxic
        sub_X_train, sub_X_test, sub_y_train, sub_y_test = train_test_split(sub_X, sub_y, 
                                                                            test_size=0.2, 
                                                                            random_state=41,
                                                                            stratify=sub_y)
        sub_X_train, sub_X_valid, sub_y_train, sub_y_valid = train_test_split(sub_X_train, sub_y_train, 
                                                                            test_size=0.25, 
                                                                            random_state=41, 
                                                                            stratify=sub_y_train)
        sub_corpus = sub_X_train.comment.values
        sub_vectorizer.fit(sub_corpus)
        sub_X_train_vec = sub_vectorizer.transform(sub_corpus).toarray()
        sub_X_valid_vec = sub_vectorizer.transform(sub_X_valid.comment.values).toarray()
        sub_X_test_vec = sub_vectorizer.transform(sub_X_test.comment.values).toarray()
        sub_y_train = sub_y_train.values
        sub_y_valid = sub_y_valid.values
        sub_y_test = sub_y_test.values
        sub_lg.fit(sub_X_train_vec, sub_y_train)
        sub_y_train_pred = sub_lg.predict(sub_X_train_vec)
        sub_y_valid_pred = sub_lg.predict(sub_X_valid_vec)
        sub_y_test_pred = sub_lg.predict(sub_X_test_vec)
        sub_f1_train.append(f1_score(sub_y_train, sub_y_train_pred))
        sub_f1_valid.append(f1_score(sub_y_valid, sub_y_valid_pred))
        sub_f1_test.append(f1_score(sub_y_test, sub_y_test_pred))

    return sub_f1_train, sub_f1_valid, sub_f1_test