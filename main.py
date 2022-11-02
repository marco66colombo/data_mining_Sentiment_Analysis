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



def main():
    df = parse.parseInput()
    print(df.describe())
    df2 = df[df['Class'] != 2]

    #print(df2['Anootated tweet'])
    df2 = df2.dropna()
    print('is nan:')
    print(df2['text'].isna().sum())

    print(tabulate(df2, headers='keys'))

    features = df2.drop("Class", axis=1)
    labels = df2["Class"]



    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.8, random_state=1)





if __name__ == '__main__':
    main()
