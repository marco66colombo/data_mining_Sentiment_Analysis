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
import pandas as pd
import sklearn.metrics as m
import sklearn.metrics
from imblearn.under_sampling import RandomUnderSampler
import nltk
import textblob
from imblearn.over_sampling import SMOTE
import seaborn as sns

from parse_date import translate_day_of_week


# compute the performance measures
def scores(y_test, y_pred, model_name):
    print('\n' + model_name + ' results:\n')
    score1 = accuracy_score(y_test, y_pred)
    print("accuracy:   %0.3f" % score1)

    print(classification_report(y_test, y_pred))

    print("confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    print('\n------------------------------\n')


def main():

    path = 'training-Obama-Romney-tweets.xlsx'
    df_raw_obama = pd.read_excel(path, sheet_name="Obama")
    df_raw_romney = pd.read_excel(path, sheet_name="Romney")

    print('Obama df description ', df_raw_obama.describe())
    print('Romney df description ', df_raw_romney.describe())

    df_raw_obama = df_raw_obama.dropna()
    df_raw_romney = df_raw_romney.dropna()

    # build single dataframe with all the examples
    ###### CHANGE HERE TO RUN THE CLASSIFIERS WITH DIFFERENT DATASETS
    df_raw = pd.concat([df_raw_obama, df_raw_romney], axis=0)
    #df_raw = df_raw_obama
    #df_raw = df_raw_romney

    # ensure that only the desired columns are used
    df_raw = df_raw[['date', 'time', 'text', 'Class']]

    # consider only examples in which the class is defined and it belongs to {-1, 0, 1}
    df = df_raw[df_raw['Class'].isin((-1, 0, 1))]

    df['Class'] = df['Class'].apply(int)

    # balance number if instances for each class
    # tried to balance number of classes but performance not enhanced
    '''rus = RandomUnderSampler(sampling_strategy='not minority', random_state=1)
    df_x, df_y = rus.fit_resample(df[['date', 'time', 'text']], df['Class'])
    df = pd.concat([df_x, df_y], axis=1)'''

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
    # print(tabulate(df, headers='keys'))


    #create new features -> USE ONE HOT ENCODING

    df['date'] = df['date'].dt.dayofweek.values

    time_one_hot = pd.get_dummies(df['time'], prefix='time')
    day_of_week_one_hot = pd.get_dummies(df['date'], prefix='date')

    df = df.drop(columns=['time', 'date'])
    #df = pd.concat([df, time_one_hot, day_of_week_one_hot], axis=1)
    df = pd.concat([df, time_one_hot], axis=1)

    labels = df["Class"].to_numpy()
    features = df.drop(columns=['Class'])  # df[['date', 'time', 'text']]

    #TODO creare una funzione che dato un test set in input, fa preprocessing e poi classifica

    # creating the training set and the test set
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=1)

    # cast to int the y_test
    y_test = [int(y_test[i]) for i in range(y_test.shape[0])]

    # compute the tf.idf representation of the texts
    vectorizer = TfidfVectorizer()
    tfidf_train = vectorizer.fit_transform(X_train['text'])
    tfidf_test = vectorizer.transform(X_test['text'])

    '''smt = SMOTE()
    tfidf_train, y_train = smt.fit_resample(tfidf_train, y_train)'''

    # add new features to the dataframe

    df_train = X_train.drop(columns=['text'])
    df_test = X_test.drop(columns=['text'])

    csr_matrix_train = csr_matrix(df_train.astype(pd.SparseDtype("float64", 0)).sparse.to_coo())
    csr_matrix_test = csr_matrix(df_test.astype(pd.SparseDtype("float64", 0)).sparse.to_coo())

    final_train = sp.sparse.hstack([tfidf_train, csr_matrix_train], 'csr')
    final_test = sp.sparse.hstack([tfidf_test, csr_matrix_test], 'csr')

    '''
    feature_names = vectorizer.get_feature_names()
    dense = tfidf_train.todense()
    lst1 = dense.tolist()
    data_train = pd.DataFrame(lst1, columns=feature_names)
    data_train['time__'] = X_train['time'].values
    data_train['date__'] = X_train['date'].dt.dayofweek.values

    dense_test = tfidf_test.todense()
    lst2 = dense_test.tolist()
    data_test = pd.DataFrame(lst2, columns=feature_names)
    data_test['time__'] = X_test['time'].values
    data_test['date__'] = X_test['date'].dt.dayofweek.values

    # same as data_train/test but with sparse matrix
    final_train = sp.sparse.hstack([tfidf_train, sp.sparse.csr_matrix(X_train['time'].values).T], 'csr')
    final_test = sp.sparse.hstack([tfidf_test, sp.sparse.csr_matrix(X_test['time'].values).T], 'csr')'''

    # SVM --------------------------------------------------------------------------------------------------------------
    # train the SVM classificator
    start = time.time()
    svm_regressor = svm.SVC(kernel='rbf', gamma=0.58, C=0.81, class_weight='balanced')
    svm_regressor.fit(tfidf_train, y_train)
    stop = time.time()
    print(f"Training time SVM: {stop - start}s")

    # compute the prediction
    y_pred = svm_regressor.predict(tfidf_test)

    # compute the performance measures
    scores(y_test, y_pred, 'SVM')

    # RANDOM FOREST ----------------------------------------------------------------------------------------------------
    # train the random forest classificator
    start = time.time()
    regressor = RandomForestRegressor(n_estimators=100, random_state=0)
    regressor.fit(final_train, y_train)
    stop = time.time()
    print(f"Training time Random Forest: {stop - start}s")

    #compute the prediction
    y_pred = regressor.predict(final_test)

    print('predicitons', y_pred)

    # convert y_pred in an array with the predicted class instead of the probability
    y_pred_1 = []
    for i in range(y_pred.shape[0]):
        if y_pred[i] < -0.2:
            y_pred_1.append(-1)
        elif y_pred[i] > 0.2:
            y_pred_1.append(1)
        else:
            y_pred_1.append(0)

    # print(y_pred_1)

    # compute the performance measures
    scores(y_test, y_pred_1, 'Random Forest')

    # NAIVE BAYES ------------------------------------------------------------------------------------------------------
    # Naive Bayes Classificator
    y = y_train
    y = y.astype(int)

    start = time.time()
    naive_bayes_classifier = MultinomialNB()
    naive_bayes_classifier.fit(final_train, y)
    stop = time.time()
    print(f"Training time Naive Bayes: {stop - start}s")

    y_pred = naive_bayes_classifier.predict(final_test)

    # compute the performance measures
    scores(y_test, y_pred, 'Naive Bayes')


if __name__ == '__main__':
    main()
