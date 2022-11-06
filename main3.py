import pandas as pd
from keras.layers import LSTM, Dense, Embedding
from keras.models import Sequential
from keras.datasets import imdb
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.utils import pad_sequences
from imblearn.under_sampling import RandomUnderSampler

import preprocess
import nltk


def main3():
    path = 'training-Obama-Romney-tweets.xlsx'
    df_raw = pd.read_excel(path, sheet_name="Romney")
    print(df_raw.describe())

    df_raw = df_raw[['date', 'time', 'text', 'Class']]
    df = df_raw[df_raw['Class'].isin((1,-1,0))]
    df = df.dropna()
    df = preprocess.makeTextCleaning(df)
    df = preprocess.makeDateCleaning(df)

    print(df.describe())
    #print(tabulate(df, headers='keys'))

    df = df.dropna()
    print(df.describe())

    features = df[['date', 'time', 'text']]
    labels = df["Class"].to_numpy()

    #TODO creare una funzione che dato un test set in input, fa preprocessing e poi classifica

    #X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=1)

    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=20000)

    X_train = pad_sequences(X_train, maxlen=100)
    X_test = pad_sequences(X_test, maxlen=100)

    model = Sequential()
    model.add(Embedding(20000, 128))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))


    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(X_train, y_train,
              batch_size=32,
              epochs=10,
              verbose=2,
              validation_data=(X_test, y_test))

    score, accuracy = model.evaluate(X_test, y_test, batch_size=32, verbose=2)

    predictions = model.predict(X_test[:5])




