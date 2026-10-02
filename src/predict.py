"""Predict sentiment of new posts with the trained SVM model (Chapter 6, Step 7).

Usage:
    python -m src.predict "I love this phone!"      # one post
    python -m src.predict                           # interactive mode
"""
import os
import sys

import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import preprocess  # noqa: E402

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "outputs", "sentiment_svm.joblib")
EMOJI = {"Positive": "😊", "Neutral": "😐", "Negative": "😠"}
_model = None


def load_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("Model not found. Run:  python -m src.train")
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_one(text: str) -> dict:
    model = load_model()
    clean = preprocess(text)
    if not clean:                                   # nothing left after cleaning
        return {"text": text, "sentiment": "Neutral", "confidence": 0.0,
                "probabilities": {}, "emoji": EMOJI["Neutral"], "note": "No usable words found"}
    probs = model.predict_proba([clean])[0]
    classes = list(model.classes_)
    best = int(probs.argmax())
    return {"text": text, "sentiment": classes[best], "confidence": round(float(probs[best]), 4),
            "probabilities": {c: round(float(p), 4) for c, p in zip(classes, probs)},
            "emoji": EMOJI[classes[best]]}


def main():
    if len(sys.argv) > 1:
        posts = [" ".join(sys.argv[1:])]
    else:
        print("Social Media Sentiment Analyzer (SVM + TF-IDF). Empty line to quit.")
        posts = iter(lambda: input("Post> ").strip(), "")
    for post in posts:
        r = predict_one(post)
        print(f"  -> {r['sentiment']} {r['emoji']}  (confidence {r['confidence']:.2f})")


if __name__ == "__main__":
    main()
