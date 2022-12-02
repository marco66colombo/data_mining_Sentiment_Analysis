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


def preprocess_data(df_raw):
    # ensure that only the desired columns are used
    df_raw = df_raw[['date', 'time', 'text', 'Class']]

    # consider only examples in which the class is defined and it belongs to {-1, 0, 1}
    df = df_raw[df_raw['Class'].isin((-1, 0, 1))]

    df['Class'] = df['Class'].apply(int)

    # print info about class distribution
    print('counts', df['Class'].value_counts())

    # apply preprocessing
    df = preprocess.makeTextCleaning(df)
    df = preprocess.makeDateCleaning(df)

    # drop na to ensure that no nan are present
    df = df.dropna()

    # describe dataframe after preprocessing
    print('df description after preprocessing ', df.describe())
    print('counts after preprocessing ', df['Class'].value_counts())
    print(tabulate(df, headers='keys'))

    # create new features -> USE ONE HOT ENCODING
    df['date'] = df['date'].dt.dayofweek.values

    time_one_hot = pd.get_dummies(df['time'], prefix='time')
    day_of_week_one_hot = pd.get_dummies(df['date'], prefix='date')

    df = df.drop(columns=['time', 'date'])
    df = pd.concat([df, time_one_hot, day_of_week_one_hot], axis=1)
    df = pd.concat([df, time_one_hot], axis=1)

    labels = df["Class"].to_numpy()
    features = df.drop(columns=['Class'])  # df[['date', 'time', 'text']]

    return features, labels

path = 'training-Obama-Romney-tweets.xlsx'
df_raw_obama = pd.read_excel(path, sheet_name="Obama")
df_raw_romney = pd.read_excel(path, sheet_name="Romney")

print('Obama df description ', df_raw_obama.describe())
print('Romney df description ', df_raw_romney.describe())

df_raw_obama = df_raw_obama.dropna()
df_raw_romney = df_raw_romney.dropna()

# build single dataframe with all the examples
# CHANGE HERE TO RUN THE CLASSIFIERS WITH DIFFERENT DATASETS
df_raw_all = pd.concat([df_raw_obama, df_raw_romney], axis=0)
df_raw_obama = df_raw_obama
df_raw_romney = df_raw_romney

features_all, labels_all = preprocess_data(df_raw_all)
features_obama, labels_obama = preprocess_data(df_raw_obama)
features_romney, labels_romney = preprocess_data(df_raw_romney)


#OBAMA
# compute the tf.idf representation of the texts
vectorizer_obama = TfidfVectorizer()
tfidf_train_obama = vectorizer_obama.fit_transform(features_obama['text'])
pickle.dump(vectorizer_obama, open("tfidf_vectorizer_obama", "wb"))
features_obama = features_obama.drop(columns=['text'])
csr_matrix_train = csr_matrix(features_obama.astype(pd.SparseDtype("float64", 0)).sparse.to_coo())
final_train = sp.sparse.hstack([tfidf_train_obama, csr_matrix_train], 'csr')

# all features
# LOGISTIC REGRESSION ----------------------------------------------------------------------------------------------
lr_classifier = LogisticRegression(random_state=0, max_iter=1000).fit(final_train, labels_obama)
filename = 'finalized_model_obama_all_feature.sav'
pickle.dump(lr_classifier, open(filename, 'wb'))

# only text
# SVM --------------------------------------------------------------------------------------------------------------
# train the SVM classificator
start = time.time()
svm_regressor_obama = svm.SVC(kernel='rbf', gamma=0.6, C=0.8, class_weight='balanced')
svm_regressor_obama.fit(tfidf_train_obama, labels_obama)
stop = time.time()
print(f"Training time SVM: {stop - start}s")

# save the model to disk
filename = 'finalized_model_obama.sav'
pickle.dump(svm_regressor_obama, open(filename, 'wb'))


# ROMNEY
# compute the tf.idf representation of the texts
vectorizer_romney = TfidfVectorizer()
tfidf_train_romney = vectorizer_romney.fit_transform(features_romney['text'])
pickle.dump(vectorizer_romney, open("tfidf_vectorizer_romney", "wb"))


# SVM --------------------------------------------------------------------------------------------------------------
# train the SVM classificator
start = time.time()
svm_regressor_romney = svm.SVC(kernel='rbf', gamma=0.6, C=0.8, class_weight='balanced')
svm_regressor_romney.fit(tfidf_train_romney, labels_romney)
stop = time.time()
print(f"Training time SVM: {stop - start}s")

# save the model to disk
filename = 'finalized_model_romney.sav'
pickle.dump(svm_regressor_romney, open(filename, 'wb'))



