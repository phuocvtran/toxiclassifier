from vncorenlp import VnCoreNLP
import json
import string
import re


def init_tokenizer():
    tokenizer = VnCoreNLP("VnCoreNLP/VnCoreNLP-1.1.1.jar",
                          annotators="wseg", max_heap_size='-Xmx500m')

    return tokenizer


def tokenize(text, tokenizer):
    sentences = tokenizer.tokenize(text)
    res = []
    for sentence in sentences:
        res.append(" ".join(sentence))

    return " ".join(res)


def transform_abbreviations(text, abbreviations):
    res = []
    words = text.split()
    for word in words:
        temp = word
        if temp in abbreviations:
            res.append(abbreviations[temp])
            continue

        temp = remove_special_char(temp)
        if temp in abbreviations:
            res.append(abbreviations[temp])
            continue

        temp = remove_duplicated_char(temp)
        if temp in abbreviations:
            res.append(abbreviations[temp])
            continue

        res.append(remove_duplicated_char(word))

    return " ".join(res)


def remove_special_char(text):
    res = re.sub(r"[^\w\s]", "", text)

    return res


def remove_duplicated_char(text):
    res = re.sub(r"[^\w\W]|(.)(?=\1)", "", text)

    return res


def remove_stop_words(text, stop_words):
    res = []
    words = text.split()
    for word in words:
        if word not in stop_words:
            res.append(word)

    return " ".join(res)


def remove_punctuation(text):
    return " ".join([word for word in text.split() if word not in string.punctuation])


def remove_url(text):
    res = re.sub(r"http\S+", "", text)

    return res


# https://stackoverflow.com/a/49146722/330558
def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def preprocess_text(text, tokenizer, abbreviations, stop_words):
    text = " ".join(text.split())
    text = text.lower()
    text = remove_emoji(text)
    text = remove_url(text)
    text = transform_abbreviations(text, abbreviations)
    text = tokenize(text, tokenizer)
    text = remove_punctuation(text)
    text = remove_stop_words(text, stop_words)
    res = text

    return res


def preprocess_df(df):
    tokenizer = init_tokenizer()
    with open("data/abbreviations.json", "r") as json_file:
        abbreviations = json.load(json_file)

    with open("data/stop_words.txt") as f:
        stop_words = f.read().splitlines()
    df.comment = df.comment.apply(preprocess_text,
                                  tokenizer=tokenizer,
                                  abbreviations=abbreviations,
                                  stop_words=stop_words)

    return df
