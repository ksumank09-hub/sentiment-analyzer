"""Original prototype (Appendix A of the report): lexicon-based sentiment with TextBlob.

Requires:  pip install textblob
Run     :  python original/sentiment_textblob.py
"""
from textblob import TextBlob

positive_words = {
    "love": "❤️", "happy": "😊", "great": "👍", "awesome": "🔥", "fun": "🎉",
    "cool": "😎", "excited": "🤩", "win": "🏆", "laugh": "😂", "joy": "😁"
}

negative_words = {
    "sad": "😢", "angry": "😠", "cry": "😭", "bad": "👎", "hate": "💔",
    "tired": "😴", "bored": "😐", "annoyed": "😤", "fail": "💥", "mad": "😡"
}

neutral_words = {
    "okay": "😐", "fine": "😶", "normal": "🫥", "average": "📊", "meh": "😑",
    "regular": "🔁", "usual": "📆", "standard": "📏", "typical": "🧾"
}


def get_emojis_by_sentiment(text):
    words = text.lower().split()
    emojis = set()
    for word in words:
        if word in positive_words:
            emojis.add(positive_words[word])
        elif word in negative_words:
            emojis.add(negative_words[word])
        elif word in neutral_words:
            emojis.add(neutral_words[word])
    return " ".join(emojis) if emojis else "No emoji match 🤔"


def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        sentiment = "Positive 😊"
    elif polarity < 0:
        sentiment = "Negative 😠"
    else:
        sentiment = "Neutral 😐"
    emojis = get_emojis_by_sentiment(text)
    return sentiment, polarity, emojis


def main():
    print("=" * 50)
    print("    SOCIAL MEDIA SENTIMENT ANALYZER")
    print("=" * 50)
    num_posts = int(input("How many posts do you want to analyze? "))
    results = []
    for i in range(num_posts):
        print(f"\nPost #{i+1}:")
        text = input("Enter the social media post: ")
        sentiment, polarity, emojis = analyze_sentiment(text)
        results.append((text, sentiment, polarity, emojis))

    print("\n" + "=" * 50)
    print("              ANALYSIS RESULTS")
    print("=" * 50)
    for i, (text, sentiment, polarity, emojis) in enumerate(results, 1):
        print(f"\nPost #{i}")
        print(f"Text: {text}")
        print(f"Sentiment: {sentiment}")
        print(f"Polarity Score: {polarity}")
        print(f"Emoji Tags: {emojis}")
    print("\nThank you for using the Sentiment Analyzer! 🙌")


if __name__ == "__main__":
    main()
