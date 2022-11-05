import pandas as pd
import preprocess
import re
from nltk.tokenize.treebank import TreebankWordDetokenizer
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras import regularizers
from keras import *
from keras.models import Sequential
from keras import layers
from keras import regularizers
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import Embedding

def depure_data(data):

    #Removing URLs with a regular expression
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    data = url_pattern.sub(r'', data)

    # Remove Emails
    data = re.sub('\S*@\S*\s?', '', data)

    # Remove new line characters
    data = re.sub('\s+', ' ', data)

    # Remove distracting single quotes
    data = re.sub("\'", "", data)

    return data

def sent_to_words(sentences, gensim=None):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))


def detokenize(text):
    return TreebankWordDetokenizer().detokenize(text)


def main():

    path = 'training-Obama-Romney-tweets.xlsx'
    df_with_class_2 = pd.read_excel(path, sheet_name="Obama")
    print(df_with_class_2.describe())

    df = df_with_class_2[df_with_class_2['Class'].isin((1,-1,0))]
    df = df.dropna()
    df = preprocess.makeTextCleaning(df)
    df = preprocess.makeDateCleaning(df)

    temp = []
    #Splitting pd.Series to list

    train = df

    data_to_list = train['text'].values.tolist()
    for i in range(len(data_to_list)):
        temp.append(depure_data(data_to_list[i]))
    data_words = list(sent_to_words(temp))
    data = []
    for i in range(len(data_words)):
        data.append(detokenize(data_words[i]))
    print(data[:5])



    max_words = 5000
    max_len = 200

    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(data)
    sequences = tokenizer.texts_to_sequences(data)
    tweets = pad_sequences(sequences, maxlen=max_len)
    print(tweets)


    embedding_layer = Embedding(1000, 64)


    model1 = Sequential()
    model1.add(layers.Embedding(max_words, 20)) #The embedding layer
    model1.add(layers.LSTM(15,dropout=0.5)) #Our LSTM layer
    model1.add(layers.Dense(3,activation='softmax'))


    model1.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])

    checkpoint1 = ModelCheckpoint("best_model1.hdf5", monitor='val_accuracy', verbose=1,save_best_only=True, mode='auto', period=1,save_weights_only=False)
    history = model1.fit(X_train, y_train, epochs=70,validation_data=(X_test, y_test),callbacks=[checkpoint1])

    model0 = Sequential()
    model0.add(layers.Embedding(max_words, 15))
    model0.add(layers.SimpleRNN(15,return_sequences=True))
    model0.add(layers.SimpleRNN(15))
    model0.add(layers.Dense(3,activation='softmax'))







if __name__ == '__main__':
    main()
