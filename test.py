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

path = 'training-Obama-Romney-tweets.xlsx'
df_raw_test = pd.read_excel(path, sheet_name="Obama")

df_raw = df_raw_test

# ensure that only the desired columns are used
# TODO NOI USIAMO TEXT, MA ERA ANOOTED TWEET, DIPENDE DA CHE FILE CI DARA'
#df_raw = df_raw['text'] # 'text' -> 'Anootated tweet'
df_raw = df_raw[['text', 'Class']]  # 'text' -> 'Anootated tweet'

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

# SVM --------------------------------------------------------------------------------------------------------------
# save the model to disk
filename_svm = 'finalized_model.sav'

# load the model from disk
loaded_model = pickle.load(open(filename_svm, 'rb'))
y_pred = loaded_model.predict(tfidf_test)

df['predicted_class'] = y_pred.tolist()
df.to_excel("output.xlsx")

# TODO DA TOGLIERE MA E' SOLO PER VEDERE SE FUNZIONA, ACCURACY ALTA PERCHE STESSI DATI DEL TRAINING
scores(np.asarray(df['Class']), np.asarray(df['predicted_class']), 'test_finale')


