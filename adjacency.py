"""
Regression model to handle adjacency.
"""

import nltk
from sklearn.linear_model import LogisticRegression

class AdjacencyModel:

    def __init__(self, adjacent, non_adjacent):
        x = [value for value in non_adjacent + adjacent]
        y = [False for _ in non_adjacent] + [True for _ in adjacent]
        reg = LogisticRegression()
        reg.fit(x, y)
        self.reg = reg

    def proba(self, observed):
        return self.reg.predict_proba([observed])[0][1]

def score(text1, text2):
    """
    Score the similarity between two texts.
    """
    return [word_similarity(text1, text2), bigram_similarity(text1, text2)]

def word_similarity(text1, text2):
    tokens1 = set(token.lower() for token in nltk.word_tokenize(text1) if any(c.isalpha() for c in token))
    tokens2 = set(token.lower() for token in nltk.word_tokenize(text2) if any (c.isalpha() for c in token))
    return len(tokens1.intersection(tokens2)) / len(tokens1.union(tokens2))

def bigram_similarity(text1, text2):
    tokens1 = list(token.lower() for token in nltk.word_tokenize(text1) if any(c.isalpha() for c in token))
    tokens2 = list(token.lower() for token in nltk.word_tokenize(text2) if any (c.isalpha() for c in token))
    bigrams1 = set(nltk.ngrams(tokens1, 2))
    bigrams2 = set(nltk.ngrams(tokens2, 2))
    return len(bigrams1.intersection(bigrams2)) / max(len(bigrams1.union(bigrams2)), 1)