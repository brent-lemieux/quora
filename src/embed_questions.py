import pandas as pd
import numpy as np

import pickle
import logging
from string import punctuation, printable

from gensim.models import Word2Vec

vec_model = Word2Vec.load('~/projects/quora/models/wordvec2')

def clean_string(q):
    cleaned = ''
    for char in q:
        if char not in punctuation and char in printable:
            cleaned += char
    return cleaned


data_path = '~/projects/quora_data'
df = pd.read_csv('{}/train.csv'.format(data_path))
df.dropna(inplace=True)

df['question1'] = df['question1'].map(clean_string)
df['question1'] = df['question1'].map(lambda x: x.lower().split())

df['question2'] = df['question2'].map(clean_string)
df['question2'] = df['question2'].map(lambda x: x.lower().split())

def q_to_mat(question, word_vectorizer):
    mat = np.zeros((30,30))
    for i, word in enumerate(question):
        if i < 30 and word in word_vectorizer.wv.vocab:
            mat[i,:] = word_vectorizer[word]
    return mat

def featurize(df):
    features = np.zeros((df.shape[0], 1800))
    for i in range(0,df.shape[0]):
        print('Question pair: {} featurized'.format(i))
        features[i,:] = np.concatenate((df['mat1'].iloc[i].flatten(), df['mat2'].iloc[i].flatten()))
    return features

if __name__ == '__main__':
    df['mat1'] = [q_to_mat(q, vec_model) for q in df['question1']]
    df['mat2'] = [q_to_mat(q, vec_model) for q in df['question2']]
    feature_mat = featurize(df)
    feature_mat.to_pickle('../../quora_data/embeddings.pkl')
