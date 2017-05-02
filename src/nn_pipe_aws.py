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

from keras.models import Sequential
from keras.layers import Dense
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

vocab_size = 10000
embed_size = 32

def remove_stop(text):
    stop_free = []
    for word in text.split():
        if word not in stop_words:
            stop_free.append(word)
    return ' '.join(stop_free)

def preprocess(f_path):
    # Load cleaned data
    df = pd.read_csv(f_path)
    df.fillna('blank', inplace=True)

    q1 = df['question1'].map(remove_stop).values
    q2 = df['question2'].map(remove_stop).values
    y = df['is_duplicate'].values
    # Vectorize text
    tokenizer = Tokenizer(num_words=vocab_size,
        lower=True, split=" ")
    tokenizer.fit_on_texts(np.concatenate((q1, q2)))
    vec1 = tokenizer.texts_to_sequences(q1)
    vec2 = tokenizer.texts_to_sequences(q2)
    # Pad sequences
    vec1 = sequence.pad_sequences(vec1, maxlen=60, dtype='int32',
        padding='pre', truncating='pre', value=0.)
    vec2 = sequence.pad_sequences(vec2, maxlen=60, dtype='int32',
        padding='pre', truncating='pre', value=0.)
    return np.concatenate((vec1, vec2), axis=1), y

# Build model

def build_mlp():
    model = Sequential()
    model.add(Embedding(vocab_size, embed_size, input_length=120))
    model.add(Flatten())
    model.add(Dense(250, activation='relu'))
    model.add(Dense(150, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def build_cnn():
    model = Sequential()
    model.add(Embedding(vocab_size, embed_size, input_length=120))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(250, activation='relu'))
    model.add(Dense(150, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Train model

def train_model(model_builder, X_train, X_test, y_train, y_test):
    model = model_builder()
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2,
        batch_size=128, verbose=1)
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    return model



if __name__ == '__main__':
    X, y = preprocess('/home/ubuntu/data/train_cleaned.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    mlp = train_model(build_mlp, X_train, X_test, y_train, y_test)
    probas = mlp.predict_proba(X_test)
    print("\n mlp log loss = ", log_loss(y_test, probas))
    cnn = train_model(build_cnn, X_train, X_test, y_train, y_test)
    probas = cnn.predict_proba(X_test)
    print("\n cnn log loss = ", log_loss(y_test, probas))
