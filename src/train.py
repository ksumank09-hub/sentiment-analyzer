"""
Train and evaluate sentiment classifiers (Chapter 6, Steps 3-6).

Pipeline : raw text -> preprocess -> TF-IDF -> classifier
Models   : Multinomial Naive Bayes, Logistic Regression, Linear SVM (main model)
Outputs  : outputs/metrics.json, outputs/*.png, outputs/sentiment_svm.joblib
"""
import json
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import preprocess  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "social_media_sentiment.csv")
OUT = os.path.join(ROOT, "outputs")
LABELS = ["Negative", "Neutral", "Positive"]
COLORS = {"Positive": "#2e9e5b", "Neutral": "#e0a800", "Negative": "#d64545"}
os.makedirs(OUT, exist_ok=True)


def make_tfidf():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)


def build_models():
    return {
        "Naive Bayes": MultinomialNB(alpha=0.5),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=2.0),
        # Calibration gives probabilities ("confidence") for the SVM.
        "SVM (linear)": CalibratedClassifierCV(LinearSVC(C=1.0), cv=5),
    }


def plot_class_distribution(df):
    counts = df["sentiment"].value_counts().reindex(LABELS)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90,
              colors=[COLORS[l] for l in counts.index])
    ax[0].set_title("Sentiment distribution (pie)")
    ax[1].bar(counts.index, counts.values, color=[COLORS[l] for l in counts.index])
    ax[1].set_title("Sentiment distribution (bar)")
    ax[1].set_ylabel("Number of posts")
    for i, v in enumerate(counts.values):
        ax[1].text(i, v + 10, str(v), ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "class_distribution.png"), dpi=150)
    plt.close()


def plot_top_words(df):
    """Top words per class. Uses a word cloud if the wordcloud package exists."""
    try:
        from wordcloud import WordCloud
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, lab in zip(axes, ["Positive", "Neutral", "Negative"]):
            text = " ".join(df.loc[df.sentiment == lab, "clean"])
            wc = WordCloud(width=500, height=350, background_color="white").generate(text)
            ax.imshow(wc)
            ax.set_title(lab, color=COLORS[lab])
            ax.axis("off")
        name = "wordclouds.png"
    except ImportError:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for ax, lab in zip(axes, ["Positive", "Neutral", "Negative"]):
            words = pd.Series(" ".join(df.loc[df.sentiment == lab, "clean"]).split())
            top = words.value_counts().head(10)[::-1]
            ax.barh(top.index, top.values, color=COLORS[lab])
            ax.set_title(f"Top words: {lab}")
        name = "top_words.png"
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, name), dpi=150)
    plt.close()
    return name


def plot_confusion(y_true, y_pred, title, fname):
    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred, labels=LABELS),
                           display_labels=LABELS).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, fname), dpi=150)
    plt.close()


def plot_model_comparison(results):
    names = list(results)
    metrics = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(names))
    w = 0.2
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, m in enumerate(metrics):
        vals = [results[n][m] for n in names]
        bars = ax.bar(x + (i - 1.5) * w, vals, w, label=m.capitalize())
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.1)
    ax.set_title("Model comparison on the 20% test set (macro-averaged)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4, frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "model_comparison.png"), dpi=150)
    plt.close()


def main():
    df = pd.read_csv(DATA).dropna(subset=["text", "sentiment"])
    df["clean"] = df["text"].apply(preprocess)
    df = df[df["clean"].str.len() > 0].reset_index(drop=True)
    print(f"Loaded {len(df)} posts\n{df.sentiment.value_counts().to_string()}\n")

    plot_class_distribution(df)
    top_words_file = plot_top_words(df)

    # 80 / 20 stratified split (Chapter 6, Step 4)
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["sentiment"], test_size=0.20, random_state=42, stratify=df["sentiment"])

    results, reports, pipes = {}, {}, {}
    for name, clf in build_models().items():
        pipe = Pipeline([("tfidf", make_tfidf()), ("clf", clf)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        p, r, f, _ = precision_recall_fscore_support(y_test, pred, average="macro", zero_division=0)
        results[name] = {"accuracy": accuracy_score(y_test, pred),
                         "precision": p, "recall": r, "f1": f}
        reports[name] = classification_report(y_test, pred, labels=LABELS, output_dict=True,
                                              zero_division=0)
        pipes[name] = (pipe, pred)
        print(f"== {name}\n{classification_report(y_test, pred, labels=LABELS, zero_division=0)}")

    plot_model_comparison(results)
    svm_pipe, svm_pred = pipes["SVM (linear)"]
    plot_confusion(y_test, svm_pred, "Confusion matrix - SVM", "confusion_matrix_svm.png")

    # 5-fold cross validation of the main model (on the full data)
    cv_pipe = Pipeline([("tfidf", make_tfidf()),
                        ("clf", CalibratedClassifierCV(LinearSVC(C=1.0), cv=3))])
    cv = cross_val_score(cv_pipe, df["clean"], df["sentiment"], cv=5, scoring="accuracy")
    print(f"SVM 5-fold CV accuracy: {cv.mean():.4f} +/- {cv.std():.4f}")

    # Most influential words for the plain linear SVM (interpretability)
    tfidf = make_tfidf().fit(X_train)
    lin = LinearSVC(C=1.0).fit(tfidf.transform(X_train), y_train)
    feats = np.array(tfidf.get_feature_names_out())
    top_terms = {cls: feats[np.argsort(lin.coef_[i])[-8:][::-1]].tolist()
                 for i, cls in enumerate(lin.classes_)}

    # Save the final model: refit on ALL data for deployment
    final = Pipeline([("tfidf", make_tfidf()),
                      ("clf", CalibratedClassifierCV(LinearSVC(C=1.0), cv=5))])
    final.fit(df["clean"], df["sentiment"])
    joblib.dump(final, os.path.join(OUT, "sentiment_svm.joblib"))

    metrics = {
        "n_posts": int(len(df)),
        "class_counts": df.sentiment.value_counts().to_dict(),
        "train_size": int(len(X_train)), "test_size": int(len(X_test)),
        "models": results,
        "svm_report": reports["SVM (linear)"],
        "svm_confusion_matrix": confusion_matrix(y_test, svm_pred, labels=LABELS).tolist(),
        "labels": LABELS,
        "svm_cv_accuracy_mean": float(cv.mean()), "svm_cv_accuracy_std": float(cv.std()),
        "top_terms_per_class": top_terms,
        "top_words_figure": top_words_file,
    }
    with open(os.path.join(OUT, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print("\nSaved model, metrics and charts to", OUT)


if __name__ == "__main__":
    main()
