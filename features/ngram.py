from collections import Counter
import nltk
import sys

from helpers import prune_counter, is_training

def train(feature, texts):

    n = feature.get("length") or 1
    threshold = feature.get("threshold") or None # Freuqency threshold for n-grams
    top = feature.get("top") or None

    # Get n-gram counter for each text
    counters = {}
    cumulative = Counter()
    for i, identifier in enumerate(texts):
        print(f"    Computing n-grams for identifier {i + 1} of {len(texts)}...")
        sys.stdout.write("\033[F")
        ngram_type = feature.get("type") or "text"
        text = texts[identifier].get(ngram_type)
        counters[identifier] = ngrams(text, n, ngram_type)
        if is_training(identifier):
            cumulative += counters[identifier]

    # Limit cumulative counter
    if threshold is not None:
        prune_counter(cumulative, threshold)
    cumulative = cumulative.most_common(top)
    print(cumulative)

    vectors = {}
    for i, identifier in enumerate(texts):
        print(f"    Computing vector for identifier {i + 1} of {len(texts)}...")
        sys.stdout.write("\033[F")
        vectors[identifier] = compute_vector(counters[identifier], cumulative)
    return vectors

def ngrams(text, n, ngram_type):
    if ngram_type == "text":
        words = nltk.word_tokenize(text.lower())
    elif ngram_type in ["paths", "cpfws"]: # paths are stored one per line
        words = text.split("\n")
    else:
        words = text.split() # for POS or CCG tagging, just split on spaces

    try:
        ngrams = list(nltk.ngrams(words, n))
    except RuntimeError:
        ngrams = []
    return Counter(ngrams)

def compute_vector(counter, cumulative):
    length = sum(counter.values())
    if length == 0:
        return [0 for _ in cumulative]
    return [counter[ngram] / length for ngram, _ in cumulative]
