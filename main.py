from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV

import parse
import openpyxl
import pandas as pd
import xlrd
from tabulate import tabulate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer



def vectorize(data,tfidf_vect_fit):
    X_tfidf = tfidf_vect_fit.transform(data['text'])
    words = tfidf_vect_fit.get_feature_names()
    X_tfidf_df = pd.DataFrame(X_tfidf.toarray())
    X_tfidf_df.columns = words
    return(X_tfidf_df)


def print_results(results):
    print('BEST PARAMS: {}\n'.format(results.best_params_))

    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, results.cv_results_['params']):
        print('{} (+/-{}) for {}'.format(round(mean, 3), round(std * 2, 3), params))


def main():
    df = parse.parseInput()
    print(df.describe())
    df2 = df[df['Class'] != 2]

    #print(df2['Anootated tweet'])
    df2 = df2.dropna()
    print('is nan:')
    print(df2['text'].isna().sum())

    features = df2.drop("Class", axis=1)
    labels = df2["Class"]

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.90, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.5, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    print(
        "Data distribution:\n- Train: {} \n- Validation: {} \n- Test: {}".format(len(y_train), len(y_val), len(y_test)))

    v = TfidfVectorizer()
    x = v.fit_transform(X_train['text'])
    X_train = vectorize(X_train, x)

    tfidf_vect = TfidfVectorizer()
    tfidf_vect_fit = tfidf_vect.fit(X_train['text'])
    X_train = vectorize(X_train, tfidf_vect_fit)

    rf = RandomForestClassifier()
    scores = cross_val_score(rf,X_train,y_train.values.ravel(),cv=5)

    rf = RandomForestClassifier()
    parameters = {
        'n_estimators': [5, 50, 100],
        'max_depth': [2, 10, 20, None]
    }

    cv = GridSearchCV(rf, parameters)
    cv.fit(X_train, y_train.values.ravel())
    print_results(cv)

    cv.best_estimator_

    X_val = vectorize(X_val, tfidf_vect_fit)

    rf1 = RandomForestClassifier(n_estimators=100, max_depth=20)
    rf1.fit(X_train, y_train.values.ravel())
    rf2 = RandomForestClassifier(n_estimators=100, max_depth=None)
    rf2.fit(X_train, y_train.values.ravel())
    rf3 = RandomForestClassifier(n_estimators=5, max_depth=None)
    rf3.fit(X_train, y_train.values.ravel())

    for mdl in [rf1, rf2, rf3]:
        y_pred = mdl.predict(X_val)
        accuracy = round(accuracy_score(y_val, y_pred), 3)
        precision = round(precision_score(y_val, y_pred), 3)
        recall = round(recall_score(y_val, y_pred), 3)
        print('MAX DEPTH: {} / # OF EST: {} -- A: {} / P: {} / R: {}'.format(mdl.max_depth,
                                                                             mdl.n_estimators,
                                                                             accuracy,
                                                                             precision,
                                                                             recall))

        X_test = vectorize(X_test['text'], tfidf_vect_fit)

        y_pred = rf2.predict(X_test)
        accuracy = round(accuracy_score(y_test, y_pred), 3)
        precision = round(precision_score(y_test, y_pred), 3)
        recall = round(recall_score(y_test, y_pred), 3)
        print('MAX DEPTH: {} / # OF EST: {} -- A: {} / P: {} / R: {}'.format(rf3.max_depth,
                                                                             rf3.n_estimators,
                                                                             accuracy,
                                                                             precision,
                                                                             recall))








if __name__ == '__main__':
    main()
