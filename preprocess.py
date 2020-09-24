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


def remove_unknown_words(text, vocabs):
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


def remove_stop_words(text, stop_words):
    res = []
    words = text.split()
    for word in words:
        if word not in stop_words:
            res.append(word)

    return " ".join(res)


def preprocess_text(text, tokenizer, abbreviations, vocabs, stop_words):
    text = " ".join(text.split())
    text = text.lower()
    text = transform_abbreviations(text, abbreviations)
    text = tokenize(text, tokenizer)
    text = remove_unknown_words(text, vocabs)
    text = remove_stop_words(text, stop_words)
    res = text

    return res


def preprocess_df(df):
    tokenizer = init_tokenizer()
    with open("data/abbreviations.json", "r") as json_file:
        abbreviations = json.load(json_file)
    
    with open("data/vietnam74K.txt") as f:
        vietnam74K = f.read().splitlines()
    vocabs = [i.lower() for i in vietnam74K]
    
    with open("data/stop_words.txt") as f:
        stop_words = f.read().splitlines()
    df.comment = df.comment.apply(preprocess_text, 
                                  tokenizer=tokenizer, 
                                  abbreviations=abbreviations, 
                                  vocabs=vocabs, 
                                  stop_words=stop_words)

    return df
