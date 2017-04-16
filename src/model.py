import pandas as pd
import numpy as np
from string import printable, punctuation

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import recall_score, precision_score, accuracy_score
from sklearn.model_selection import train_test_split


def clean(string):
    string = ''.join([char for char in string if char in printable and char not in punctuation])
    return string.lower()

def vectorize(q1, q2, grams=[1,1]):
    all_questions = np.concatenate((q1.values, q2.values))
    tf = TfidfVectorizer(ngram_range=grams, min_df=.001, stop_words='english').fit(all_questions)
    print('vectorized')
    mat1 = tf.transform(q1.values).todense()
    mat2 = tf.transform(q2.values).todense()
    mat = np.hstack((mat1, mat2))
    return mat


if __name__ == '__main__':
    data_path = '/Users/blemieux/projects/quora_data'
    df = pd.read_csv('{}/train.csv'.format(data_path))
    df.dropna(inplace=True)
    train_examples = 100000
    df = df.iloc[:train_examples,:]
    q1, q2, dup = df['question1'], df['question2'], df['is_duplicate'].values
    print('cleaning data...')
    q1 = q1.map(clean)
    q2 = q2.map(clean)
    print('vectorizing text...')
    mat = vectorize(q1, q2)
    X_train, X_test, y_train, y_test = train_test_split(mat, dup)
    mod = RandomForestClassifier()
    print('training model...')
    mod.fit(X_train, y_train)
    preds = mod.predict(X_test)
    print ('recall', recall_score(y_test, preds))
    print ('precision', precision_score(y_test, preds))
    print ('accuracy', accuracy_score(y_test, preds))
