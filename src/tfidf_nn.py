'''
* Load in data
* Clean and lemmatize
* Bag of words vectorizer
* Pad sequences
* Keras models with embeddings --> MLP/CNN
'''

import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as stop_words
from sklearn.feature_extraction.text import TfidfVectorizer

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

vocab_size = 50000


def preprocess(f_path):
    # Load cleaned data
    df = pd.read_csv(f_path)
    df.fillna('blank', inplace=True)
    q1 = df['question1'].values.reshape(len(df), 1)
    q2 = df['question2'].values.reshape(len(df), 1)
    X = np.concatenate((q1, q2), axis=1)
    y = df['is_duplicate'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    # Vectorize text
    tf = TfidfVectorizer()
    tf.fit(np.concatenate((X_train[:,0], X_train[:,1])))
    train_vec1 = tf.transform(X_train[:,0]).todense()
    train_vec2 = tf.transform(X_train[:,1]).todense()
    test_vec1 = tf.transform(X_test[:,0]).todense()
    test_vec2 = tf.transform(X_test[:,1]).todense()
    train_mats = np.array([np.array([v1, v2]) for v1, v2 in zip(train_vec1, train_vec2)])
    return train_mats

# Build model

def build_mlp():
    model = Sequential()
    model.add(Dense(350, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Train model

def train_model(model_builder, X_train, X_test, y_train, y_test, num_epochs=2):
    model = model_builder()
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=num_epochs,
        batch_size=500, verbose=1)
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    return model



if __name__ == '__main__':
    train_mats = preprocess('../../quora_data/train_cleaned.csv')
