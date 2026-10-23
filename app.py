
import pickle
from flask import Flask, request, jsonify
from prometheus_flask_exporter import PrometheusMetrics

app=Flask(__name__)

metrics = PrometheusMetrics(app)

with open("news_model.pkl", "rb") as file:
    news_model = pickle.load(file)


@app.route("/")
def home():
    return "Welcome to the Fake News Detection API!"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    s_c = data["source_credibility"]
    a_c = data["author_credibility"]
    c_s = data["clickbait_score"]
    s_s = data["sentiment_score"]
    a_l = data["article_length"]
    n_q = data["num_quotes"]
    n_i = data["num_images"]
    n_l = data["num_links"]
    social = data["social_shares"]
    v_s = data["verified_source"]

    input_data = [[
        s_c, a_c, c_s, s_s, a_l,
        n_q, n_i, n_l, social, v_s
    ]]

    prediction = news_model.predict(input_data)

    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
