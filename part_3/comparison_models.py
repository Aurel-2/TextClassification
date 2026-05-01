from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

import pandas as pd
import pprint


# Naive Bayes
def train_nb(X_train, X_test, y_train, y_test):
    model = MultinomialNB()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, pred),
        "f1": f1_score(y_test, pred, average="weighted"),
    }


# Logistic Regression
def train_logistic_regression(X_train, X_test, y_train, y_test):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, pred),
        "f1": f1_score(y_test, pred, average="weighted"),
    }

# Decision Tree
def train_decision_tree(X_train, X_test, y_train, y_test):
    model = DecisionTreeClassifier(max_depth=20, random_state=42)
    model.fit(X_train, y_train)   # FIX ICI
    pred = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, pred),
        "f1": f1_score(y_test, pred, average="weighted"),
    }



# Random Forest
def train_random_forest(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, pred),
        "f1": f1_score(y_test, pred, average="weighted"),
    }

# SVM
def train_svm(X_train, X_test, y_train, y_test):
    model = SVC(kernel="linear", C=1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)


    return {
        "accuracy": accuracy_score(y_test, pred),
        "f1": f1_score(y_test, pred, average="weighted"),
    }



# TF-IDF
def vectorize_text(X_train, X_test):
    vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 2))

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    return X_train_vec, X_test_vec


def spam_dataset():
    df = pd.read_csv('../data/spam.csv', sep="\t", header=None, quoting=3)

    y = df.iloc[:, 0]
    X = df.iloc[:, 1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_vec, X_test_vec = vectorize_text(X_train, X_test)

    results = {
        'decision_tree': train_decision_tree(X_train_vec, X_test_vec, y_train, y_test),
        'random_forest': train_random_forest(X_train_vec, X_test_vec, y_train, y_test),
        'logistic_regression': train_logistic_regression(X_train_vec, X_test_vec, y_train, y_test),
        'naive_bayes': train_nb(X_train_vec, X_test_vec, y_train, y_test),
        'SVM': train_svm(X_train_vec, X_test_vec, y_train, y_test),
    }

    return results


def imdb_dataset():
    df1 = pd.read_csv('../data/imdb_train.csv')
    df2 = pd.read_csv('../data/imdb_test.csv')

    X_train = df1['text']
    y_train = df1['label']

    X_test = df2['text']
    y_test = df2['label']

    X_train_vec, X_test_vec = vectorize_text(X_train, X_test)

    results = {
        'decision_tree': train_decision_tree(X_train_vec, X_test_vec, y_train, y_test),
        'random_forest': train_random_forest(X_train_vec, X_test_vec, y_train, y_test),
        'logistic_regression': train_logistic_regression(X_train_vec, X_test_vec, y_train, y_test),
        'naive_bayes': train_nb(X_train_vec, X_test_vec, y_train, y_test),
        'SVM': train_svm(X_train_vec, X_test_vec, y_train, y_test),
    }

    return results




def main():
    all_results = {
        'IMDB': imdb_dataset(),
        'SPAM': spam_dataset(),
    }

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(all_results)

if __name__ == '__main__':
    main()