from sklearn import model_selection, __all__
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tabulate import tabulate
import preprocess
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import sklearn.metrics as m
import sklearn.metrics


def load_as_list(X, y):
    documents = X.values.tolist()
    labels = y.values.tolist()

    return documents, labels



def main():

    path = 'training-Obama-Romney-tweets.xlsx'
    df_raw = pd.read_excel(path, sheet_name="Obama")
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

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=1)

    #X_train = features
    #y_train = labels

    documents = X_train['text']
    labels = y_train

    vectorizer = TfidfVectorizer()
    tfidf_train = vectorizer.fit_transform(documents)
    tfidf_test = vectorizer.transform(X_test['text'])

    feature_names = vectorizer.get_feature_names()
    dense = tfidf_train.todense()
    lst1 = dense.tolist()
    data_train = pd.DataFrame(lst1, columns=feature_names)
    data_train['time'] = X_train['time'].values
    data_train['date'] = X_train['date'].dt.dayofweek.values


    dense_test = tfidf_test.todense()
    lst2 = dense_test.tolist()
    data_test = pd.DataFrame(lst2, columns=feature_names)
    data_test['time'] = X_test['time'].values
    data_test['date'] = X_test['date'].dt.dayofweek.values
    #df[‘ScheduledDay_dayofweek’] = df[‘ScheduledDay’].dt.dayofweek


    regressor = RandomForestRegressor(n_estimators=100, random_state=0)
    regressor.fit(data_train, labels)

    y_pred = regressor.predict(data_test)
    #y_pred = model_selection.cross_val_predict(regressor, data_train, labels, cv=10)

    print(y_pred)
    y_pred_1 = []
    for i in range(y_pred.shape[0]):
        if y_pred[i] < -0.2:
            y_pred_1.append(-1)
        elif y_pred[i] > 0.2:
            y_pred_1.append(1)
        else:
            y_pred_1.append(0)


    print(y_pred_1)
    #print(type(y_test))

    y_test_1 = []
    for i in range(y_test.shape[0]):
        y_test_1.append(int(y_test[i]))

    '''y_test_1 = []
    for i in range(labels.shape[0]):
        y_test_1.append(int(labels[i]))'''

    acc = m.accuracy_score(y_test_1, y_pred_1)
    cm = m.confusion_matrix(y_test_1, y_pred_1)

    print('accuracy: ', acc)
    print('cm: ', cm)


if __name__ == '__main__':
    main()
