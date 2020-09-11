import json
import re
from underthesea import word_tokenize, ner


def trim(sentence):
    return " ".join(sentence.split())


def remove_abbreviation(sentence):
    abbreviation_path = "preproc/abbreviation.json"
    word_list = sentence.split()
    with open(abbreviation_path) as f:
        abbreviation = json.load(f)

    for index, word in enumerate(word_list):
        if word.lower() in abbreviation:
            word_list[index] = abbreviation[word.lower()]

    return " ".join(word_list)


def remove_long_char(sentence):
    longest_word = len("nghiêng")
    word_list = sentence.split()
    for index, word in enumerate(word_list):
        if len(word) > longest_word:
            del word_list[index]
    
    return " ".join(word_list)


def remove_special_char(sentence):
    return re.sub(r"[^\w\s]", "", sentence)


def remove_number(sentence):
    return re.sub(r"[0-9]", "", sentence)


def remove_link(sentence):
    return re.sub(r"(?:(?:https?|ftp|file):\/\/|www\.|ftp\.)(?:\([-A-Z0-9+&@#\/%=~_|$?!:,.]*\)|[-A-Z0-9+&@#\/%=~_|$?!:,.])*(?:\([-A-Z0-9+&@#\/%=~_|$?!:,.]*\)|[A-Z0-9+&@#\/%=~_|$])", "", sentence)


def remove_name_entity(sentence):
    tag_list = ner(sentence)
    name_list = []
    for index, content in enumerate(tag_list):
        if content[1] == "Np" and content[2] == "B-NP":
            name_list.append(content[0])
    
    for name in name_list:
        sentence = sentence.replace(name, "")

    return sentence


def tokenzie(sentence):
    return word_tokenize(sentence, format="text")


def remove_stop_word(sentence):
    stop_word_path = "preproc/stop_word.txt"
    word_list = sentence.split()
    with open(stop_word_path) as f:
        stop_word = f.read().split("\n")

    for index, word in enumerate(word_list):
        if word.lower() in stop_word:
            del word_list[index]

    return " ".join(word_list)


def remove_duplicated_char(sentence):
    word_list = sentence.split()

    for index, word in enumerate(word_list):
        word_list[index] = re.sub(r"[^\w\s]|(.)(?=\1)", "", word, flags=re.IGNORECASE)
    
    return " ".join(word_list)


def preproc(df):
    cp_df = df.copy()
    for index, row in cp_df.iterrows():
        sentence = row.comment
        sentence = trim(sentence)
        sentence = remove_link(sentence)
        sentence = remove_abbreviation(sentence)
        sentence = remove_special_char(sentence)
        sentence = remove_abbreviation(sentence)
        sentence = remove_duplicated_char(sentence)
        sentence = remove_abbreviation(sentence)
        sentence = remove_name_entity(sentence)
        sentence = remove_abbreviation(sentence)
        sentence = remove_number(sentence)
        sentence = remove_abbreviation(sentence)
        sentence = trim(sentence)
        sentence = remove_long_char(sentence)
        sentence = tokenzie(sentence)
        sentence = remove_stop_word(sentence)
        if len(sentence) == 0:
            sentence = "empty_string"
        sentence = sentence.lower()
        cp_df.at[index, 'comment'] = sentence

    return cp_df


def new_preproc(df):
    cp_df = df.copy()
    for index, row in cp_df.iterrows():
        sentence = row.comment
        sentence = trim(sentence)
        sentence = remove_link(sentence)
        sentence = remove_abbreviation(sentence)
        sentence = remove_special_char(sentence)
        sentence = remove_abbreviation(sentence)
        sentence = remove_duplicated_char(sentence)
        sentence = remove_abbreviation(sentence)
        sentence = remove_number(sentence)
        sentence = remove_abbreviation(sentence)
        sentence = trim(sentence)
        sentence = remove_long_char(sentence)
        sentence = tokenzie(sentence)
        sentence = remove_stop_word(sentence)
        if len(sentence) == 0:
            sentence = "empty_string"
        sentence = sentence.lower()
        cp_df.at[index, 'comment'] = sentence

    return cp_df