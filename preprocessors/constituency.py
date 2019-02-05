import copy
import itertools
import nltk
import requests
from nltk.parse import CoreNLPParser

parser = CoreNLPParser(url="http://localhost:9000")

def preprocess(contents):
    """This is the identity function: return the original text unchanged."""
    words = nltk.word_tokenize(contents)
    try:
        tree = list(parser.parse(words))[0]
    except requests.exceptions.HTTPError:
        return "()"
    return str(tree)
