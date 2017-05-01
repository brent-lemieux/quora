from keras.models import Sequential
from keras.layers import Dense

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, accuracy_score

def create_model():
    model = Sequential()
    model.add(Dense(350, input_dim=1800, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return model

def train_chunks(model, epochs=10):
    master_x_test = np.zeros((1,1800))
    master_y_test = np.zeros((1,1))
    for epoch in range(epochs):
        X = pd.read_csv("embeddings.csv", iterator=True, chunksize=1000)
        y = pd.read_csv("labels.csv", iterator=True, chunksize=1000)
        print('epoch == {}'.format(epoch))
        for x_chunk, y_chunk in zip(X, y):
            x_chunk, y_chunk = x_chunk.values[:,1:], y_chunk.values[:,1:]
            X_train, X_test, y_train, y_test = train_test_split(x_chunk, y_chunk, random_state=5)
            model.train_on_batch(X_train, y_train)
            if epoch == 0:
                master_x_test = np.concatenate((master_x_test, X_test), axis=0)
                master_y_test = np.concatenate((master_y_test, y_test), axis=0)
            del x_chunk, X_train, X_test, y_chunk, y_train, y_test
        del X, y
    preds = model.predict(master_x_test)
    preds = [round(pred) for pred in preds.flatten()]
    print(accuracy_score(master_y_test, preds), precision_score(master_y_test, preds), recall_score(master_y_test, preds))
    return model, master_x_test, master_y_test





if __name__ == '__main__':
    # data = pd.read_csv("embeddings.csv", iterator=True, chunksize=1000)
    # labels = pd.read_csv("labels.csv", iterator=True, chunksize=1000)
    mlp = create_model()
    mlp, x_test, y_test = train_chunks(mlp, epochs=5)
    # preds = mlp.predict(x_test)
    # probas = mlp.predict_proba(x_test)



    # pickle.dump(mlp, open('net.pkl', 'wb'))
