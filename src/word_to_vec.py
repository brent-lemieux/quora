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

def clean_string(q):
    cleaned = ''
    for char in q:
        if char not in punctuation and char in printable:
            cleaned += char
    return cleaned

# change for AWS Instance
data_path = '/home/ubuntu/quora_data'
df = pd.read_csv('{}/train.csv'.format(data_path))
df.dropna(inplace=True)

sentences = pd.concat((df['question1'], df['question2']))
sentences = sentences.map(clean_string)
sentences = sentences.map(lambda x: x.lower().split())

size = sentences.map(len)

vec_model = Word2Vec(sentences, size=60, workers=3)
vec_model.save('/home/ubuntu/quora/models/wordvec')
