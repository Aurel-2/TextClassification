import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def get_models():
    return {
        "svm": SVC(kernel="linear", C=1),
        "logistic_regression": LogisticRegression(max_iter=1000),
        "naive_bayes": MultinomialNB(),
        "decision_tree": DecisionTreeClassifier(max_depth=20, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

def train_and_save_all(X_train, X_test, y_train, y_test, vectorizer, dataset_name):
    os.makedirs("models", exist_ok=True)
    results = {}
    joblib.dump(vectorizer, f"models/tfidf_{dataset_name}.pkl")
    for name, model in get_models().items():
        trained_model, scores = train_model(model, X_train, X_test, y_train, y_test)
        joblib.dump(trained_model, f"models/{name}_{dataset_name}.pkl")
        results[name] = scores

    return results


def train_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    scores = {
        "accuracy": accuracy_score(y_test, pred),
        "f1": f1_score(y_test, pred, average='weighted')
    }
    return model, scores


def vectorize_text(X_train, X_test):
    vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    return X_train_vec, X_test_vec, vectorizer


def spam_model():
    df = pd.read_csv('../data/spam.csv', sep="\t", header=None, quoting=3)

    y = df.iloc[:, 0]
    X = df.iloc[:, 1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train_vec, X_test_vec, vectorizer = vectorize_text(X_train, X_test)
    train_and_save_all(X_train_vec, X_test_vec, y_train, y_test, vectorizer, "spam")

def imdb_model():
    df1 = pd.read_csv('../data/imdb_train.csv')
    df2 = pd.read_csv('../data/imdb_test.csv')
    X_train = df1['text']
    y_train = df1['label']
    X_test = df2['text']
    y_test = df2['label']
    X_train_vec, X_test_vec, vectorizer = vectorize_text(X_train, X_test)
    train_and_save_all(X_train_vec, X_test_vec, y_train, y_test, vectorizer, "imdb")

if __name__ == "__main__":
    print("Training and saving models...")
    print("Spam models")
    spam_model()
    print("IMDB models")
    imdb_model()
    print("Done")