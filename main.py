from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV
from tabulate import tabulate

import preprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import string
import re
import csv

import nltk

def load_as_list(X, y):
    documents = X.values.tolist()
    labels = y.values.tolist()

    return documents, labels



def main():

    path = 'training-Obama-Romney-tweets.xlsx'
    df_with_class_2 = pd.read_excel(path, sheet_name="Obama")
    print(df_with_class_2.describe())

    df = df_with_class_2[df_with_class_2['Class'] != 2]
    df = df.dropna()
    preprocess.makeTextCleaning(df)
    #print(tabulate(df, headers='keys'))

    features = df.drop("Class", axis=1)
    labels = df["Class"].to_numpy()

    #TODO creare una funzione che dato un test set in input, fa preprocessing e poi classifica

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.8, random_state=1)



    #documents, labels = load_as_list(X_train['text'], y_train)
    documents = X_train['text'].to_numpy()
    labels = y_train

    #labels = labels.astype(np.float)

    print('ciao', type(labels[12]))
    vectorizer = TfidfVectorizer()
    tfidf_train = vectorizer.fit_transform(documents)

    tfidf_test = vectorizer.transform(X_test['text'])

    regressor = RandomForestRegressor(n_estimators=20, random_state=0)
    regressor.fit(tfidf_train, labels)

    y_pred = regressor.predict(tfidf_test)











'''
# Function: vectorize_test, see project statement for more details
# vectorizer: A trained TFIDF vectorizer
# user_input: A string of arbitrary length
# Returns: A sparse TFIDF representation of the input string of shape (1, X), dtype: scipy.sparse.csr.csr_matrix
#
# This function computes the TFIDF representation of the input string, using
# the provided TfidfVectorizer.
def vectorize_test(vectorizer, user_input):
    # Initialize the TfidfVectorizer model and document-term matrix
    tfidf_test = None
    processed_input = preprocessing(user_input)
    single_input = []
    single_input.append(processed_input)
    tfidf_test = vectorizer.transform(single_input)
    return tfidf_test


# Function: train_nb_model(training_documents, training_labels)
# training_data: A sparse TfIDF document-term matrix, dtype: scipy.sparse.csr.csr_matrix
# training_labels: A list of integers (0 or 1)
# Returns: A trained model
def train_nb_model(training_data, training_labels):
    # Initialize the GaussianNB model and the output label
    nb_model = GaussianNB()

    # Write your code here.  You will need to make use of the GaussianNB fit()
    # function.  You probably need to transfrom your data into a dense numpy array.
    X_train = training_data.toarray()
    Y_train = training_labels
    nb_model = nb_model.fit(X_train, Y_train)
    return nb_model

# Function: get_model_prediction(nb_model, tfidf_test)
# nb_model: A trained GaussianNB model
# tfidf_test: A sparse TFIDF representation of the input string of shape (1, X), dtype: scipy.sparse.csr.csr_matrix
# Returns: A predicted label for the provided test data (int, 0 or 1)
def get_model_prediction(nb_model, tfidf_test):
    # Initialize the output label
    label = 0

    # Write your code here.  You will need to make use of the GaussianNB
    # predict() function. You probably need to transfrom your data into a dense numpy array.
    # [YOUR CODE HERE]
    X_test = tfidf_test.toarray()
    label = nb_model.predict(X_test)
    return label




# Function: instantiate_models()
# This function does not take any input
# Returns: Three instantiated machine learning models
#
# This function instantiates the three imported machine learning models, and
# returns them for later downstream use.  You do not need to train the models
# in this function.
def instantiate_models():
    logistic = None
    svm = None
    mlp = None

    # [YOUR CODE HERE]
    logistic = LogisticRegression(random_state=100)
    svm = LinearSVC(random_state=100)
    mlp = MLPClassifier(random_state=100)

    return logistic, svm, mlp




# Function: test_model(model, word2vec, training_documents, training_labels)
# model: An instantiated machine learning model
# word2vec: A pretrained Word2Vec model
# test_data: A list of test documents
# test_labels: A list of integers (all 0 or 1)
# Returns: Precision, recall, F1, and accuracy values for the test data
#
# This function tests an input machine learning model by extracting features
# for each preprocessed test document and then predicting an output label for
# that document.  It compares the predicted and actual test labels and returns
# precision, recall, f1, and accuracy scores.
def test_model(model, word2vec, test_documents, test_labels):
    precision = None
    recall = None
    f1 = None
    accuracy = None

    # Write your code here:
    processed_documents = []
    for doc in test_documents:
        processed_documents.append(string2vec(word2vec, doc))
    X_test = np.array(processed_documents)
    Y_test = model.predict(X_test)

    accuracy = accuracy_score(test_labels, Y_test)
    precision = precision_score(test_labels, Y_test)
    recall = recall_score(test_labels, Y_test)
    f1 = f1_score(test_labels, Y_test)

    return precision, recall, f1, accuracy


# -------------------------- New in Project Part 3! --------------------------
# Function: count_words(user_input)
# user_input: A string of arbitrary length
# Returns: An integer value
#
# This function counts the number of words in the input string.
def count_words(user_input):
    num_words = 0
    # [YOUR CODE HERE]
    tokens = nltk.tokenize.word_tokenize(user_input)
    tokens_filtered = []
    for token in tokens:
        # match = p.match(token)
        if token not in string.punctuation:
            tokens_filtered.append(token)

    num_words = len(tokens_filtered)
    return num_words

# Function: words_per_sentence(user_input)
# user_input: A string of arbitrary length
# Returns: A floating point value
#
# This function computes the average number of words per sentence
def words_per_sentence(user_input):
    wps = 0.0
    # [YOUR CODE HERE]
    sentences = nltk.tokenize.sent_tokenize(user_input)
    # print(sentences)
    num_words = 0
    for sentence in sentences:
        num_words += count_words(sentence)
    wps = float(num_words) / float(len(sentences))
    # print(wps)
    return wps

'''
'''

# Use this main function to test your code. Sample code is provided to assist with the assignment,
# feel free to change/remove it. In project components, this function might be graded, see rubric for details.
if __name__ == "__main__":

    # Set things up ahead of time by training the TfidfVectorizer and Naive Bayes model
    documents, labels = load_as_list("dataset.csv")



    # Instantiate and train the machine learning models
    logistic, svm, mlp = instantiate_models()

    logistic = train_model(logistic, word2vec, documents, labels)
    svm = train_model(svm, word2vec, documents, labels)
    mlp = train_model(mlp, word2vec, documents, labels)


    # Test the machine learning models to see how they perform on the small test set provided.
    # Write a classification report to a CSV file with this information.
    # Loading the dataset
    test_documents, test_labels = load_as_list("test.csv")
    models = [logistic, svm, mlp]
    model_names = ["Logistic Regression", "SVM", "Multilayer Perceptron"]
    outfile = open("classification_report.csv", "w", newline='\n')
    outfile_writer = csv.writer(outfile)
    outfile_writer.writerow(["Name", "Precision", "Recall", "F1", "Accuracy"]) # Header row
    i = 0
    while i < len(models): # Loop through other results
        p, r, f, a = test_model(models[i], word2vec, test_documents, test_labels)
        if models[i] == None: # Models will be None if functions have not yet been implemented
            outfile_writer.writerow([model_names[i],"N/A", "N/A", "N/A", "N/A"])
        else:
            outfile_writer.writerow([model_names[i], p, r, f, a])
        i += 1
    outfile.close()
'''


if __name__ == '__main__':
    main()
