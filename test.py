import time

import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix
from sklearn import model_selection, __all__, svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from tabulate import tabulate
import preprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import sklearn.metrics as m
import sklearn.metrics
from imblearn.under_sampling import RandomUnderSampler
import nltk
import textblob
from imblearn.over_sampling import SMOTE
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal, uniform

from main import scores
from parse_date import translate_day_of_week
import pickle

def test_function(df_raw, name):

    print('Length of dataframe before preprocessing: ', len(df_raw))
    df_raw['id'] = df_raw.index + 1

    # ensure that only the desired columns are used
    # TODO NOI USIAMO TEXT, MA ERA ANOOTED TWEET, DIPENDE DA CHE FILE CI DARA'
    # df_raw = df_raw['text'] # 'text' -> 'Anootated tweet'
    df_raw = df_raw[['id', 'text', 'Class']]  # 'text' -> 'Anootated tweet'

    df_raw = df_raw.dropna()

    # TODO NON AVREMO 'CLASS' NEL FILE DI TEST QUINDI VA TOLTO
    df = df_raw[df_raw['Class'].isin((-1, 0, 1))]
    df['Class'] = df['Class'].apply(int)

    # apply preprocessing
    df = preprocess.makeTextCleaning(df)

    # drop na to ensure that no nan are present
    df = df.dropna()

    # compute the tf.idf representation of the texts
    vectorizer = pickle.load(open("tfidf_vectorizer", "rb"))

    tfidf_test = vectorizer.transform(df['text'])

    print('Length of dataframe after preprocessing: ', len(df))

    # SVM --------------------------------------------------------------------------------------------------------------
    # save the model to disk
    filename_svm = 'finalized_model.sav'

    # load the model from disk
    loaded_model = pickle.load(open(filename_svm, 'rb'))
    y_pred = loaded_model.predict(tfidf_test)

    output_file_name = "output_" + name + ".xlsx"

    df['predicted_class'] = y_pred.tolist()
    df = df.reset_index()  # make sure indexes pair with number of rows

    df.to_excel(output_file_name)

    # TODO DA TOGLIERE MA E' SOLO PER VEDERE SE FUNZIONA, ACCURACY ALTA PERCHE STESSI DATI DEL TRAINING
    scores(np.asarray(df['Class']), np.asarray(df['predicted_class']), 'test_finale')

    # print output txt
    output_file_name_txt = name + ".txt"
    f_out = open(output_file_name_txt, 'w')

    for index, row in df.iterrows():
        f_out.write(str(row['id']) + ';;' + str(row['predicted_class']) + '\n')

    f_out.close()


path = 'training-Obama-Romney-tweets.xlsx'
df_raw_obama = pd.read_excel(path, sheet_name="Obama")
# df_raw_romney = pd.read_excel(path, sheet_name="Romney")

test_function(df_raw_obama, "obama")
#test_function(df_raw_romney, "romney")




