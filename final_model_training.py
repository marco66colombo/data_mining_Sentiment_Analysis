import time
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
from parse_date import translate_day_of_week
import pickle

path = 'training-Obama-Romney-tweets.xlsx'
df_raw_obama = pd.read_excel(path, sheet_name="Obama")
df_raw_romney = pd.read_excel(path, sheet_name="Romney")

print('Obama df description ', df_raw_obama.describe())
print('Romney df description ', df_raw_romney.describe())

# build single dataframe with all the examples
# CHANGE HERE TO RUN THE CLASSIFIERS WITH DIFFERENT DATASETS
df_raw = pd.concat([df_raw_obama, df_raw_romney], axis=0)
#df_raw = df_raw_obama
#df_raw = df_raw_romney

# ensure that only the desired columns are used
df_raw = df_raw[['text', 'Class']]

df_raw = df_raw.dropna()
# consider only examples in which the class is defined and it belongs to {-1, 0, 1}
df = df_raw[df_raw['Class'].isin((-1, 0, 1))]
df['Class'] = df['Class'].apply(int)

# print info about class distribution
print('counts', df['Class'].value_counts())
print(tabulate(df, headers='keys'))
# apply preprocessing
df = preprocess.makeTextCleaning(df)

# drop na to ensure that no nan are present
df = df.dropna()

# describe dataframe after preprocessing
print('df description after preprocessing ', df.describe())
print('counts after preprocessing ', df['Class'].value_counts())
#print(tabulate(df, headers='keys'))

labels = df["Class"].to_numpy()
features = df.drop(columns=['Class'])

# compute the tf.idf representation of the texts
vectorizer = TfidfVectorizer()
tfidf_train = vectorizer.fit_transform(df['text'])
pickle.dump(vectorizer, open("tfidf_vectorizer", "wb"))

'''smt = SMOTE()
tfidf_train, y_train = smt.fit_resample(tfidf_train, y_train)'''

# SVM --------------------------------------------------------------------------------------------------------------
# train the SVM classificator
start = time.time()
svm_regressor = svm.SVC(kernel='rbf', gamma=0.58, C=0.81, class_weight='balanced')
svm_regressor.fit(tfidf_train, labels)
stop = time.time()
print(f"Training time SVM: {stop - start}s")

# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(svm_regressor, open(filename, 'wb'))





