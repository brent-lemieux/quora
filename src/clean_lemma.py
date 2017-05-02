'''This file contains functions to preprocess text in Pandas DataFrames -- run this in Python2 for now...'''

import pickle
from string import punctuation
from string import printable
import os
import spacy
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import pandas as pd


if not 'nlp' in locals():
    print("Loading English Module...")
    nlp = spacy.load('en')
    print("Module loading complete.")


def clean_text(text):
    '''remove special characters and punctuation -- also lower case all
    characters'''
    try:
        cleaned = ''
        for char in text:
            if char in printable and char not in punctuation:
                cleaned += char.lower()
        return cleaned
    except:
        return text


def lemmatize(text, stop_words=ENGLISH_STOP_WORDS):
    '''lemmatize the tweet -- returns each word to its base dictionary form'''
    try:
        text = unicode(text)
        text = nlp(text)
        tokens = [token.lemma_ for token in text if token not in stop_words and token]
        return ' '.join(tokens)
    except:
        return text


'''
* Load in data
* Clean and lemmatize
* Write to csv
'''

# Load, clean, and lemmatize

def load_clean(f_path):
    data = pd.read_csv(f_path)
    print('Cleaning question1')
    data['question1'] = data['question1'].map(clean_text)
    print('Lemmatizing question1')
    data['question1'] = data['question1'].map(lemmatize)
    print('Cleaning question2')
    data['question2'] = data['question2'].map(clean_text)
    print('Lemmatizing question2')
    data['question2'] = data['question2'].map(lemmatize)
    return data


if __name__ == '__main__':
    # train = load_clean('../../quora_data/train.csv')
    # train.to_csv('../../quora_data/train_cleaned.csv')
    test = load_clean('../../quora_data/test.csv')
    test.to_csv('../../quora_data/test_cleaned.csv')
