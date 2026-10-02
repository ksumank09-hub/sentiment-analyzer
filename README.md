# Sentiment Analysis in Social Media (Mini Project)

Classifies social-media posts as **Positive, Neutral or Negative** using NLP preprocessing,
**TF-IDF** features and a **Support Vector Machine (SVM)**. Naive Bayes and Logistic Regression are
trained as comparison models. Includes a Flask web app and the original TextBlob prototype.

## Project structure
```
app.py                       Flask web app + JSON API
templates/index.html         Web page
data/generate_dataset.py     Creates the sample dataset
data/social_media_sentiment.csv   Sample dataset (3,000 posts, columns: text, sentiment)
src/preprocess.py            Cleaning, tokenization, stop words, stemming
src/train.py                 TF-IDF + NB / LR / SVM, evaluation, charts, saves model
src/predict.py               Predict sentiment of new posts (command line)
original/sentiment_textblob.py   Original TextBlob prototype (Appendix A of the report)
outputs/                     Trained model, metrics.json, charts, screenshots, training_log.txt
report/Mini_Project_Report_Completed.docx   Updated report (new Chapter 7 sections, Appendix C and D)
```

## How to run
Python 3.9+ is required.
```
pip install -r requirements.txt
python -m src.train            # trains, evaluates, writes outputs/ (the model is already included)
python -m src.predict "I love this new phone!"
python app.py                  # open http://127.0.0.1:5000
```
Run all commands from the project root folder. `textblob` is only needed for
`original/sentiment_textblob.py`; `wordcloud` is optional (a top-words chart is drawn without it).

## Results on the included sample dataset (20% test set, 600 posts)
| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Naive Bayes | 93.8% | 93.9% | 93.9% | 93.9% |
| Logistic Regression | 94.7% | 94.7% | 94.7% | 94.7% |
| SVM (linear) | 94.7% | 94.7% | 94.7% | 94.7% |

SVM 5-fold cross-validation accuracy: 95.2% (+/- 1.3 points).

**Important:** the sample dataset is synthetic (template-based posts with about 5% deliberately wrong
labels), so ~95% is close to the maximum possible on it. These numbers show that the pipeline works;
they are **not** the accuracy you would get on real tweets.

## Using a real dataset
Save a CSV as `data/social_media_sentiment.csv` with the columns `text` and `sentiment`
(values `Positive`, `Neutral`, `Negative`), for example a Kaggle tweet dataset or Sentiment140
(which must first be converted to these three labels), then run `python -m src.train` again.

## API
`POST /api/predict` with JSON `{"posts": ["text 1", "text 2"]}` returns the label, confidence and
class probabilities for each post and a summary count.

## Limitations
English only; sarcasm and mixed opinions remain hard; emojis are removed during preprocessing
(as designed in the report); models use default hyper-parameters.
