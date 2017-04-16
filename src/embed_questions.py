# from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
# from keras.models import Model
# from keras import regularizers
# import keras

import pandas as pd
import numpy as np

import pickle
import logging
from string import punctuation, printable

from gensim.models import Word2Vec

vec_model = Word2Vec.load('/home/ubuntu/quora/models/wordvec')

def clean_string(q):
    cleaned = ''
    for char in q:
        if char not in punctuation and char in printable:
            cleaned += char
    return cleaned


data_path = '/home/ubuntu/quora_data'
df = pd.read_csv('{}/train.csv'.format(data_path))
df.dropna(inplace=True)

df['question1'] = df['question1'].map(clean_string)
df['question1'] = df['question1'].map(lambda x: x.lower().split())

df['question2'] = df['question2'].map(clean_string)
df['question2'] = df['question2'].map(lambda x: x.lower().split())

def q_to_mat(question, word_vectorizer):
    mat = np.zeros((60,60))
    for i, word in enumerate(question):
        if i < 60 and word in word_vectorizer.wv.vocab:
            mat[i,:] = word_vectorizer[word]
    return mat

def featurize(df):
    features = np.zeros((df.shape[0], 7200))
    for i in range(0,df.shape[0]):
        print('Question pair: {} featurized'.format(i))
        features[i,:] = np.concatenate((df['mat1'].iloc[i].flatten(), df['mat2'].iloc[i].flatten()))
    return features

if __name__ == '__main__':
    df['mat1'] = [q_to_mat(q, vec_model) for q in df['question1']]
    df['mat2'] = [q_to_mat(q, vec_model) for q in df['question2']]
    feature_mat = featurize(df)
    pickle.dump(feature_mat, open('/home/ubuntu/quora_data/nn_mat.pkl', 'wb'))
