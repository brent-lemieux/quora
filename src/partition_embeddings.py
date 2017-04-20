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
    '''
    Turn each question into vector form based on
    '''
    mat = np.zeros((30,30))
    for i, word in enumerate(question):
        if i < 30 and word in word_vectorizer.wv.vocab:
            mat[i,:] = word_vectorizer[word]
    return mat.flatten()

def partition_featurized(df, splits=20):
    '''
    Code for using the word to vec model to create partitions of word embeddings to be used in training a neural net.
    '''
    interval_size = int(df.shape[0] / splits)
    start_interval = 0
    for i in range(1, splits+1):
        end_interval = interval_size * i
        features = np.zeros((df.iloc[start_interval:,:].shape[0], 1800))
        for idx in range(0, df.iloc[start_interval:end_interval,:].shape[0]):
            print('Question pair: {} featurized'.format(idx))
            features[idx,:] = np.concatenate((df['mat1'].iloc[start_interval + idx], df['mat2'].iloc[start_interval + idx]))
        file_name = '../../quora_data/partitions/embeddings{}.npz'.format(i)
        np.save(file_name, features)
        start_interval = end_interval + 1


if __name__ == '__main__':
    df['mat1'] = [q_to_mat(q, vec_model) for q in df['question1']]
    df['mat2'] = [q_to_mat(q, vec_model) for q in df['question2']]
    feature_mat = partition_featurized(df)
