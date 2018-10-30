# function word frequency
from collections import Counter
import nltk

from helpers import prune_counter

def train(feature, texts):
    threshold = feature.get("threshold") or None
    # https://www.wordfrequency.info/
    words = [word.strip() for word in open(feature["filename"])]
    vectors = {}
    for identifier in texts:
        text = texts[identifier].get("text")
        vectors[identifier] = fn_freq(text, words)
    return vectors

def words(text):
    words = nltk.word_tokenize(text.lower())
    return Counter(words)

def fn_freq(text, words):
    text = text.lower().split()
    vector = [text.count(word) / len(words) for word in words]
    return vector
