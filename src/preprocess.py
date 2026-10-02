"""Text preprocessing for social media posts (Chapter 6, Step 2)."""
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Keep negations: "not good" must not become "good".
KEEP = {"not", "no", "nor", "never", "cannot", "cant", "dont", "wont"}
STOPWORDS = set(ENGLISH_STOP_WORDS) - KEEP

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")
NON_ALPHA_RE = re.compile(r"[^a-z0-9\s]")      # removes punctuation AND emojis
SPACES_RE = re.compile(r"\s+")

SLANG_MAP = {"u": "you", "ur": "your", "rly": "really", "coz": "because",
             "luv": "love", "gr8": "great", "ppl": "people", "tmrw": "tomorrow",
             "v": "very", "pls": "please", "thx": "thanks"}


def simple_stem(word: str) -> str:
    """Very light rule-based stemmer (used when NLTK is not installed)."""
    if word.endswith(("ss", "us", "is")):        # useless, bonus, analysis: keep as is
        return word
    for suffix, repl in (("ies", "y"), ("ing", ""), ("ed", ""), ("ly", ""), ("es", ""), ("s", "")):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: len(word) - len(suffix)] + repl
    return word


try:                                             # optional, better quality if available
    from nltk.stem import WordNetLemmatizer
    _lem = WordNetLemmatizer()
    _lem.lemmatize("tests")                      # raises LookupError if wordnet is missing
    def normalise(word):                         # noqa: E306
        return _lem.lemmatize(word)
except Exception:                                # nltk or its data not available
    normalise = simple_stem


def clean_text(text: str) -> str:
    """Remove URLs, @mentions, #hashtags, emojis, punctuation; lowercase.
    Digits are kept here so slang like 'gr8' can be fixed later."""
    text = str(text)
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(" ", text)
    text = text.lower()
    text = NON_ALPHA_RE.sub(" ", text)
    return SPACES_RE.sub(" ", text).strip()


def tokenize(text: str):
    return text.split()


def preprocess(text: str) -> str:
    """Full pipeline: clean -> tokenize -> slang fix -> stopwords -> stem/lemmatize."""
    tokens = tokenize(clean_text(text))
    tokens = [SLANG_MAP.get(t, t) for t in tokens]
    tokens = [t for t in tokens if not t.isdigit() and t not in STOPWORDS and len(t) > 1]
    tokens = [normalise(t) for t in tokens]
    return " ".join(tokens)


if __name__ == "__main__":
    demo = "@john I LOVE this new phone!!! 😍 #tech https://t.co/abc123 coz it's gr8"
    print("RAW  :", demo)
    print("CLEAN:", clean_text(demo))
    print("FINAL:", preprocess(demo))
