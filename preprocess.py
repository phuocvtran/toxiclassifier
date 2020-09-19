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
        temp = word
        if temp in abbreviations:
            res.append(abbreviations[temp])
            continue
        
        temp = remove_special_char(temp)
        if temp in abbreviations:
            res.append(abbreviations[remove_special_char(temp)])
            continue
            
        temp = remove_duplicated_char(temp)
        if temp in abbreviations:
            res.append(abbreviations[remove_duplicated_char(temp)])
            continue
        
        res.append(remove_duplicated_char(word))

    return " ".join(res)


def remove_unknown_words(text):
    with open("data/vietnam74K.txt") as f:
        vietnam74K = f.read().splitlines()
    vocabs = [i.lower() for i in vietnam74K]
    res = []
    words = text.split()
    for word in words:
        if " ".join(word.split("_")) in vocabs:
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
    text = tokenize(text, tokenizer)
    text = remove_unknown_words(text)
    text = remove_stop_words(text)
    res = text

    return res


def preprocess_df(df):
    tokenizer = init_tokenizer()
    df.comment = df.comment.apply(preprocess_text, tokenizer=tokenizer)

    return df