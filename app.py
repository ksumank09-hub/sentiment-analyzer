"""Flask web app for the Social Media Sentiment Analyzer.

Run:  python app.py      then open http://127.0.0.1:5000
"""
from flask import Flask, jsonify, render_template, request

from src.predict import load_model, predict_one

app = Flask(__name__)
MAX_POSTS = 200
MAX_LEN = 1000


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Body: {"posts": ["text 1", "text 2", ...]}  (or {"text": "..."})."""
    data = request.get_json(silent=True) or {}
    posts = data.get("posts") or ([data["text"]] if data.get("text") else [])
    posts = [str(p).strip()[:MAX_LEN] for p in posts if str(p).strip()][:MAX_POSTS]
    if not posts:
        return jsonify({"error": "Please enter at least one post."}), 400
    results = [predict_one(p) for p in posts]
    summary = {"Positive": 0, "Neutral": 0, "Negative": 0}
    for r in results:
        summary[r["sentiment"]] += 1
    return jsonify({"results": results, "summary": summary, "total": len(results)})


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    load_model()                      # fail early if the model has not been trained
    app.run(debug=False)
