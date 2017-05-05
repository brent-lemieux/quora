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
from keras.layers import Dense, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.layers.core import Reshape

vocab_size = 50000
embed_size = 256
question_len = 60

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
    vec1 = sequence.pad_sequences(vec1, maxlen=question_len, dtype='int32',
        padding='pre', truncating='pre', value=0.)
    vec2 = sequence.pad_sequences(vec2, maxlen=question_len, dtype='int32',
        padding='pre', truncating='pre', value=0.)
    return np.concatenate((vec1, vec2), axis=1), y

# Build model


def build_cnn():
    model = Sequential()
    model.add(Embedding(vocab_size, embed_size, input_length=question_len * 2))
    # model.add(Flatten())
    model.add(Reshape((1,embed_size*question_len,2)))
    model.add(Conv2D(filters=32, kernel_size=(2,embed_size), padding='same', activation='tanh'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Train model

def train_model(model_builder, X_train, X_test, y_train, y_test, num_epochs=2):
    model = model_builder()
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=num_epochs,
        batch_size=500, verbose=1)
    return model



if __name__ == '__main__':
    X, y = preprocess('/home/ubuntu/data/train_cleaned.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    cnn = train_model(build_cnn, X_train, X_test, y_train, y_test, num_epochs=3)
