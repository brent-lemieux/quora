import pandas as pd
import numpy as np

import pickle
import logging
from string import punctuation, printable

from gensim.models import Word2Vec

def clean_string(q):
    cleaned = ''
    for char in q:
        if char not in punctuation and char in printable:
            cleaned += char
    return cleaned

def q_to_mat(question, word_vectorizer):
    mat = np.zeros((30,30))
    for i, word in enumerate(question):
        if i < 30 and word in word_vectorizer.wv.vocab:
            mat[i,:] = word_vectorizer[word]
    return mat.flatten()

def featurize(df):
    features = np.zeros((df.shape[0], 1800))
    labels = np.zeros((df.shape[0]))
    for i in range(0,df.shape[0]):
        print('Question pair: {} featurized'.format(i))
        features[i,:] = np.concatenate((df['mat1'].iloc[i], df['mat2'].iloc[i]))
        labels[i] = df['is_duplicate'].iloc[i]
    xdf = pd.DataFrame(features)
    ydf = pd.DataFrame(labels)
    print(xdf.shape, ydf.shape)
    return xdf, ydf

if __name__ == '__main__':
    vec_model = Word2Vec.load('/home/ubuntu/wordvec2')

    data_path = '/home/ubuntu'
    df = pd.read_csv('{}/train.csv'.format(data_path))
    df.dropna(inplace=True)

    df['question1'] = df['question1'].map(clean_string)
    df['question1'] = df['question1'].map(lambda x: x.lower().split())

    df['question2'] = df['question2'].map(clean_string)
    df['question2'] = df['question2'].map(lambda x: x.lower().split())

    df['mat1'] = [q_to_mat(q, vec_model) for q in df['question1']]
    df['mat2'] = [q_to_mat(q, vec_model) for q in df['question2']]
    feature_mat, labels = featurize(df)

    feature_mat.to_csv('/home/ubuntu/embeddings.csv')
    labels.to_csv('/home/ubuntu/labels.csv')
