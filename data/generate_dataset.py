"""
Generate a labelled sample dataset of social-media style posts.

Why: the project must run offline and without API keys. For the final
submission you can replace data/social_media_sentiment.csv with a real
dataset (e.g. Sentiment140 / a Kaggle tweets dataset) that has the same two
columns: text, sentiment  (values: Positive / Negative / Neutral).

The generator mixes templates, topic words, hashtags, mentions, URLs,
emojis, slang, upper/lower casing and ~5% label noise so that the problem
is not trivially perfect.
"""
import os
import random
import re
import pandas as pd

random.seed(42)

TOPICS = ["phone", "camera", "movie", "concert", "restaurant", "app", "update",
          "laptop", "service", "delivery", "game", "hotel", "match", "series",
          "battery", "teacher", "coffee shop", "airline", "headphones", "show"]

POS_TEMPLATES = [
    "I love this {t}! Absolutely amazing",
    "The {t} is awesome and so much fun",
    "Best {t} ever, totally worth it",
    "Really happy with the {t}, great experience",
    "Wow the {t} was fantastic, highly recommend",
    "Such a brilliant {t}, made my day",
    "Super excited about the new {t}, looks incredible",
    "{t} exceeded my expectations, loving it",
    "Great job team, the {t} works perfectly",
    "Can't stop smiling, the {t} is wonderful",
    "The {t} was good and the staff were really kind",
    "This {t} is so cool, five stars",
]
NEG_TEMPLATES = [
    "I hate this {t}, absolutely terrible",
    "The {t} is awful and a complete waste of money",
    "Worst {t} ever, totally disappointed",
    "Really unhappy with the {t}, horrible experience",
    "Wow the {t} was a disaster, do not buy",
    "Such a bad {t}, ruined my day",
    "So annoyed about the {t}, it keeps failing",
    "{t} is broken again, so frustrating",
    "Terrible job, the {t} stopped working",
    "Feeling sad and angry, the {t} was horrible",
    "The {t} was bad and the support was rude",
    "This {t} is so boring and slow, zero stars",
]
NEU_TEMPLATES = [
    "The {t} is on the table near the window",
    "Just saw the {t} announcement, details tomorrow",
    "The {t} opens at nine and closes at five",
    "Reading about the {t} for my assignment",
    "There is a {t} event happening this weekend",
    "Got the {t} delivered on Tuesday",
    "The {t} is okay, nothing special",
    "It's a regular day, checking the {t} again",
    "Is the {t} available in other colours?",
    "Compared the {t} with last year's model",
    "The {t} costs about the same as before",
    "Heard people talking about the {t} today",
]
# genuinely mixed / hard cases, labelled by the dominant feeling
MIXED = [
    ("The {t} has great visuals but a weak story", "Neutral"),
    ("Loved the {t} at first but now it is a bit disappointing", "Negative"),
    ("The {t} was slow at the start but ended up really good", "Positive"),
    ("Not bad, the {t} is fine I guess", "Neutral"),
]
POS_EMOJI = ["😊", "😍", "👍", "🔥", "🎉", "❤️", "😎", "🤩"]
NEG_EMOJI = ["😠", "😡", "👎", "😢", "😭", "💔", "😤", "🤬"]
NEU_EMOJI = ["😐", "🤔", "😶", "📆", "🧾"]
TAGS = ["#tech", "#review", "#daily", "#news", "#life", "#trending", "#opinion"]
USERS = ["@john", "@priya", "@techguru", "@support", "@alex_22", "@mira"]
SLANG = {"really": "rly", "because": "coz", "you": "u", "very": "v", "great": "gr8",
         "love": "luv", "people": "ppl", "tomorrow": "tmrw"}


# Extra word-level variety so the model learns individual sentiment words,
# not just whole sentence templates.
POS_WORDS = ["amazing", "good", "great", "excellent", "perfect", "lovely", "brilliant",
             "fantastic", "superb", "better", "best", "happy", "delightful", "impressive"]
NEG_WORDS = ["bad", "worse", "worst", "awful", "terrible", "horrible", "poor", "useless",
             "disappointing", "annoying", "pathetic", "sad", "broken", "disgusting"]
NEU_WORDS = ["okay", "normal", "regular", "average", "standard", "typical", "usual",
             "available", "scheduled", "new", "different", "ordinary", "similar"]
FRAMES = [
    "The {t} is {w}",
    "{w} {t} today",
    "Honestly the {t} was {w}",
    "The {t} is {w} and the update made everything {w2}",
    "My {t} feels {w}, kind of {w2}",
    "Just tried the {t}, it is {w}",
    "The {t} was {w} and so was the staff",
]


def word_level(label):
    pool = {"Positive": POS_WORDS, "Negative": NEG_WORDS, "Neutral": NEU_WORDS}[label]
    return random.choice(FRAMES).format(t=random.choice(TOPICS), w=random.choice(pool),
                                        w2=random.choice(pool))


def decorate(text, label):
    emo = {"Positive": POS_EMOJI, "Negative": NEG_EMOJI, "Neutral": NEU_EMOJI}[label]
    if random.random() < 0.45:
        text += " " + random.choice(emo)
    if random.random() < 0.30:
        text += " " + random.choice(TAGS)
    if random.random() < 0.20:
        text = random.choice(USERS) + " " + text
    if random.random() < 0.10:
        text += " https://t.co/" + "".join(random.choices("abcxyz123", k=8))
    if random.random() < 0.25:
        for k, v in SLANG.items():
            text = re.sub(r'\b' + k + r'\b', v, text)
    if random.random() < 0.20:
        text = text.upper() if random.random() < 0.3 else text.lower()
    if random.random() < 0.10:
        text += random.choice(["!!!", "!!", "...", "??"])
    return text


def build(n_per_class=1000, noise=0.05):
    rows = []
    for label, temps in [("Positive", POS_TEMPLATES), ("Negative", NEG_TEMPLATES),
                         ("Neutral", NEU_TEMPLATES)]:
        for _ in range(n_per_class):
            if random.random() < 0.08:                    # mixed / hard sample
                t, used = random.choice(MIXED)
                text = t.format(t=random.choice(TOPICS))
            elif random.random() < 0.35:                  # word-level variety
                used = label
                text = word_level(label)
            else:
                used = label
                text = random.choice(temps).format(t=random.choice(TOPICS))
            text = decorate(text, used)
            if random.random() < noise:                   # label noise
                used = random.choice([l for l in ("Positive", "Negative", "Neutral")
                                      if l != used])
            rows.append((text, used))
    df = pd.DataFrame(rows, columns=["text", "sentiment"]).sample(frac=1, random_state=42)
    return df.reset_index(drop=True)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    df = build()
    df.to_csv(os.path.join(here, "social_media_sentiment.csv"), index=False, encoding="utf-8")
    print(df.sentiment.value_counts())
    print(df.head(8).to_string())
