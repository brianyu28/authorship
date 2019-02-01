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
        return ""
    all_paths = paths(tree)
    result = ""
    for path in all_paths:
        result += " ".join(path)
        result += "\n"
    return result

def paths(tree):
    if not isinstance(tree, nltk.Tree):
        return [[]]
    label = tree.label()
    if len(tree):
        all_paths = []
        for child in tree:
            all_paths += [[label] + path for path in paths(child)]
        return all_paths
    else:
        return [[label]]

