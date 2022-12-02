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

from old_version import scores
from parse_date import translate_day_of_week
import pickle

path = 'final-testData-no-label-Obama-tweets.xlsx'
df_raw_obama = pd.read_excel(path, sheet_name="Obama")
# df_raw_romney = pd.read_excel(path, sheet_name="Romney")


print('Length of dataframe obama before preprocessing: ', len(df_raw_obama))
#df_raw_obama['id'] = df_raw_obama.index + 1

# ensure that only the desired columns are used
df_raw_obama = df_raw_obama[['id', 'text']]

#df_raw = df_raw.dropna()

# apply preprocessing
df_obama = preprocess.makeTextCleaning(df_raw_obama)

# drop na to ensure that no nan are present
# df = df.dropna()

# compute the tf.idf representation of the texts
vectorizer = pickle.load(open("tfidf_vectorizer_obama", "rb"))

tfidf_test_obama = vectorizer.transform(df_obama['text'])

print('Length of dataframe obama after preprocessing: ', len(df_obama))

# SVM --------------------------------------------------------------------------------------------------------------
# save the model to disk
filename_svm = 'finalized_model_obama.sav'

# load the model from disk
loaded_model_obama = pickle.load(open(filename_svm, 'rb'))
y_pred = loaded_model_obama.predict(tfidf_test_obama)

output_file_name = "output_" + "obama" + ".xlsx"

df_obama['predicted_class'] = y_pred.tolist()
df_obama = df_obama.reset_index()  # make sure indexes pair with number of rows

df_obama.to_excel(output_file_name)

# print output txt
output_file_name_txt = "obama" + ".txt"
f_out = open(output_file_name_txt, 'w')

for index, row in df_obama.iterrows():
    f_out.write(str(row['id']) + ';;' + str(row['predicted_class']) + '\n')

f_out.close()


# ROMNEY ---------------------------------------------------------------------------------------------------------------

path = 'final-testData-no-label-Romney-tweets.xlsx'
df_raw_romney = pd.read_excel(path, sheet_name="Romney")

print('Length of dataframe romney before preprocessing: ', len(df_raw_romney))
#df_raw_romney['id'] = df_raw_romney.index + 1

# ensure that only the desired columns are used
df_raw_romney = df_raw_romney[['id', 'text']]

#df_raw = df_raw.dropna()

# apply preprocessing
df_romney = preprocess.makeTextCleaning(df_raw_romney)

# drop na to ensure that no nan are present
# df = df.dropna()

# compute the tf.idf representation of the texts
vectorizer = pickle.load(open("tfidf_vectorizer_romney", "rb"))

tfidf_test_romney = vectorizer.transform(df_romney['text'])

print('Length of dataframe romeny after preprocessing: ', len(df_romney))

# SVM --------------------------------------------------------------------------------------------------------------
# save the model to disk
filename_svm = 'finalized_model_romney.sav'

# load the model from disk
loaded_model_romney = pickle.load(open(filename_svm, 'rb'))
y_pred_romney = loaded_model_romney.predict(tfidf_test_romney)

output_file_name = "output_" + "romney" + ".xlsx"

df_romney['predicted_class'] = y_pred_romney.tolist()
df_romney = df_romney.reset_index()  # make sure indexes pair with number of rows

df_romney.to_excel(output_file_name)

# print output txt
output_file_name_txt = "romney" + ".txt"
f_out = open(output_file_name_txt, 'w')

for index, row in df_romney.iterrows():
    f_out.write(str(row['id']) + ';;' + str(row['predicted_class']) + '\n')

f_out.close()



