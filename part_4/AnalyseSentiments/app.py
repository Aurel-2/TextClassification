from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

vectorizer = joblib.load("tfidf_imdb.pkl")
models = {
    "naive_bayes": joblib.load("naive_bayes_imdb.pkl"),
    "decision_tree": joblib.load("decision_tree_imdb.pkl"),
    "logistic_regression": joblib.load("logistic_regression_imdb.pkl")
}

@app.route("/")
def home():
    return render_template("index.html")


def predict(text, model_name):
    model = models[model_name]
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    return prediction

#pour montrer les mots les plus importants pour la prédiction :
def explain(text):
    X = vectorizer.transform([text])
    scores = X.toarray()[0]
    feature_names = vectorizer.get_feature_names_out()
    indices = np.where(scores > 0)[0]
    if len(indices) == 0:
        return []
    sorted_indices = indices[np.argsort(scores[indices])[::-1]]
    top_indices = sorted_indices[:5]
    return [feature_names[i] for i in top_indices]

@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.json
    text = data.get("text")
    model_name = data.get("model")

    if not text:
        return jsonify({"error": "Aucun texte fourni"}), 400

    if model_name not in models:
        return jsonify({"error": "Modèle invalide"}), 400

    prediction = predict(text, model_name)
    explanation = explain(text)
    return jsonify({
        "model": model_name,
        "prediction": int(prediction),
        "explanation": explanation
    })

if __name__ == "__main__":
    app.run(debug=True)