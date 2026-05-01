from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.sparse import vstack
import numpy as np

# On charge les données IMDB
dataset = load_dataset("imdb")

train_data = dataset["train"].shuffle(seed=12).select(range(5000))
test_data = dataset["test"].shuffle(seed=12).select(range(5000))
unsup_data = dataset["unsupervised"].shuffle(seed=12).select(range(5000))

X_train = train_data["text"]
y_train = train_data["label"]

X_test = test_data["text"]
y_test = test_data["label"]

X_unsup = unsup_data["text"]

# On vectorise
vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
X_unsup_vec = vectorizer.transform(X_unsup)

# on crée un modèle de base supervisé
model_base = LogisticRegression(max_iter=1000)
model_base.fit(X_train_vec, y_train)

pred_base = model_base.predict(X_test_vec)
acc_base = accuracy_score(y_test, pred_base)

# on applique KMeans sur les données non étiquetées

kmeans = KMeans(n_clusters=2, random_state=12, n_init=10)
clusters = kmeans.fit_predict(X_unsup_vec)

pseudo_labels_1 = clusters

#labels inversés
pseudo_labels_2 = 1 - clusters

#On combine les données étiquetées et non étiquetées pour entraîner les modèles semi-supervisés
X_combined_1 = vstack([X_train_vec, X_unsup_vec])
y_combined_1 = np.concatenate([y_train, pseudo_labels_1])

X_combined_2 = vstack([X_train_vec, X_unsup_vec])
y_combined_2 = np.concatenate([y_train, pseudo_labels_2])

#On crée les modèles
model_semi_1 = LogisticRegression(max_iter=1000)
model_semi_1.fit(X_combined_1, y_combined_1)

pred_semi_1 = model_semi_1.predict(X_test_vec)
acc_semi_1 = accuracy_score(y_test, pred_semi_1)

# modele avec labels inversés
model_semi_2 = LogisticRegression(max_iter=1000)
model_semi_2.fit(X_combined_2, y_combined_2)

pred_semi_2 = model_semi_2.predict(X_test_vec)
acc_semi_2 = accuracy_score(y_test, pred_semi_2)


print("Précision des modèles :")
print("Supervisé seul :", acc_base, "\n")
print("Semi-supervisé :", acc_semi_1)
print("Semi-supervisé (inversé) :", acc_semi_2)