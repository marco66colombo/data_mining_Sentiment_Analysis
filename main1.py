import time
import scipy as sp
from scipy.sparse import csr_matrix
from sklearn import model_selection, __all__, svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, make_scorer, \
    balanced_accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
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
from sklearn.model_selection import StratifiedKFold
from statistics import mean
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate



def start():

    path = 'training-Obama-Romney-tweets.xlsx'
    df_raw_obama = pd.read_excel(path, sheet_name="Obama")
    df_raw_romney = pd.read_excel(path, sheet_name="Romney")

    print('Obama df description ', df_raw_obama.describe())
    print('Romney df description ', df_raw_romney.describe())

    df_raw_obama = df_raw_obama.dropna()
    df_raw_romney = df_raw_romney.dropna()

    # build single dataframe with all the examples
    # CHANGE HERE TO RUN THE CLASSIFIERS WITH DIFFERENT DATASETS
    df_raw = pd.concat([df_raw_obama, df_raw_romney], axis=0)
    #df_raw = df_raw_obama
    #df_raw = df_raw_romney

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

    # cross_validate(RandomForestRegressor(n_estimators=100, random_state=0, min_samples_split=10, min_samples_leaf=2, max_features='sqrt'), 'naive bayes', 10, features, labels, False)

    cross_validate(LogisticRegression(random_state=0, max_iter=1000), 'LogisticRegression', 10, features, labels, True)
    cross_validate(LogisticRegression(random_state=0, max_iter=1000), 'LogisticRegression', 10, features, labels, False)

    cross_validate(svm.SVC(kernel='rbf', gamma=0.6, C=0.8, class_weight='balanced'), 'svm-rbf', 10, features, labels, True)
    cross_validate(svm.SVC(kernel='rbf', gamma=0.6, C=0.8, class_weight='balanced'), 'svm-rbf', 10, features, labels, False)

    cross_validate(svm.SVC(kernel='linear', class_weight='balanced'), 'svm-linear', 10, features, labels, True)
    cross_validate(svm.SVC(kernel='linear', class_weight='balanced'), 'svm-linear', 10, features, labels, False)

    cross_validate(MultinomialNB(), 'naive bayes', 10, features, labels, True)
    cross_validate(MultinomialNB(), 'naive bayes', 10, features, labels, False)



def encodeFeatures(X_train, X_test, useonlytext):

    vectorizer = TfidfVectorizer()

    if(useonlytext):
        tfidf_train = vectorizer.fit_transform(X_train['text'])
        tfidf_test = vectorizer.transform(X_test['text'])
        return tfidf_train, tfidf_test

    else:
        # add also the day and hour features
        tfidf_train = vectorizer.fit_transform(X_train['text'])
        tfidf_test = vectorizer.transform(X_test['text'])
        df_train = X_train.drop(columns=['text'])
        df_test = X_test.drop(columns=['text'])

        csr_matrix_train = csr_matrix(df_train.astype(pd.SparseDtype("float64", 0)).sparse.to_coo())
        csr_matrix_test = csr_matrix(df_test.astype(pd.SparseDtype("float64", 0)).sparse.to_coo())

        final_train = sp.sparse.hstack([tfidf_train, csr_matrix_train], 'csr')
        final_test = sp.sparse.hstack([tfidf_test, csr_matrix_test], 'csr')

        return final_train, final_test

def compute_metrics(scores):
    n = len(scores)
    f1 = {'-1': 0, '0': 0, '1': 0}
    precision = {'-1': 0, '0': 0, '1': 0}
    recall = {'-1': 0, '0': 0, '1': 0}

    print(f1)

    for elem in scores:
        for key in f1:
            key = str(key)
            f1[key] = f1[key] + elem[key]['f1-score']
            precision[key] = precision[key] + elem[key]['precision']
            recall[key] = recall[key] + elem[key]['recall']

    for key in f1:
        f1[key] = f1[key]/n
        precision[key] = precision[key] / n
        recall[key] = recall[key] / n

    return f1, precision, recall

def cross_validate(model, modelName, cv, X, Y, useOnlyText):

    skf = StratifiedKFold(n_splits=cv, random_state=1, shuffle=True)
    accuracy = []
    scores = []

    for train_index, test_index in skf.split(X, Y):
        #print("Train:", train_index, "Test:", test_index)
        X_train = X.iloc[train_index, :]
        y_train = Y[train_index]
        X_test = X.iloc[test_index, :]
        y_test = Y[test_index]

        X_train, X_test = encodeFeatures(X_train, X_test, useOnlyText)

        start = time.time()
        model.fit(X_train, y_train)
        stop = time.time()
        print('\nModel: ', modelName)
        print(f"Training time: {stop - start}s")

        # compute the prediction
        y_pred = model.predict(X_test)
        accuracy.append(accuracy_score(y_test, y_pred))
        scores.append(classification_report(y_test, y_pred, output_dict='True'))

    print('\nModel: ', modelName, ', use only text: ', useOnlyText)
    f1, precision, recall = compute_metrics(scores)
    print('\nf1 ', f1)
    print('\nprecision', precision)
    print('\nrecall', recall)
    print('\nOverall Accuracy:', mean(accuracy) * 100, '%')


start()



