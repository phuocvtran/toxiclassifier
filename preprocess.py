from vncorenlp import VnCoreNLP
import pandas as pd
import json
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


def transform_abbreviations(text):
    with open("data/abbreviations.json", "r") as json_file:
        abbreviations = json.load(json_file)
    res = []
    words = text.split()
    for word in words:
        if word in abbreviations:
            res.append(abbreviations[word])
            continue
        
        word = remove_special_char(word)
        if word in abbreviations:
            res.append(abbreviations[remove_special_char(word)])
            continue
            
        word = remove_duplicated_char(word)
        if word in abbreviations:
            res.append(abbreviations[remove_duplicated_char(word)])
            continue
        
        res.append(word)

    return " ".join(res)


def remove_unknown_words(text):
    with open("data/single_vocab.txt") as f:
        vocabs = f.read().splitlines()
    res = []
    words = text.split()
    for word in words:
        if word in vocabs:
            res.append(word)

    return " ".join(res)


def remove_special_char(text):
    res = re.sub(r"[^\w\s]", "", text)

    return res


def remove_duplicated_char(text):
    res = re.sub(r"[^\w\W]|(.)(?=\1)", "", text)

    return res


def remove_stop_words(text):
    with open("data/stop_words.txt") as f:
        stop_words = f.read().splitlines()
    res = []
    words = text.split()
    for word in words:
        if word not in stop_words:
            res.append(word)

    return " ".join(res)


def preprocess_text(text, tokenizer):
    text = " ".join(text.split())
    text = text.lower()
    text = transform_abbreviations(text)
    text = remove_unknown_words(text)
    text = remove_stop_words(text)
    text = tokenize(text, tokenizer)
    res = text

    return res


def preprocess_df(df):
    tokenizer = init_tokenizer()
    df.comment = df.comment.apply(preprocess_text, tokenizer=tokenizer)

    return df