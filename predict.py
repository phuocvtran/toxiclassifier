from model_tox import ModelTox
from tfidf import Tfidf
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", required=True,
                    help="Tên của mô hình dùng để phân loại (lg hoặc svm).")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-t", "--text", help="Bình luận.")
group.add_argument(
    "-f", "--file", help="Đường dẫn đến file .txt chứa các bình luận. Mỗi bình luận nằm trên một dòng.")
args = vars(parser.parse_args())

if args["text"] is not None:
    text = [args["text"]]
else:
    with open(args["file"]) as f:
        text = f.read().splitlines()

vectorizer = Tfidf("res/vectorizer.joblib")
model = ModelTox(f"res/{args['model']}.joblib")

vec = vectorizer.transform(text)
pred = model.predict(vec)

for i, t in enumerate(text):
    print(f"{t}\n[Nhãn] {pred[i]}")
