import nltk
import requests
from nltk.parse import CoreNLPParser

parser = CoreNLPParser(url="http://localhost:9000")

def preprocess(contents):
    """This is the identity function: return the original text unchanged."""
    sentences = nltk.sent_tokenize(contents)
    trees = []
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        try:
            tree = list(parser.parse(words))[0]
            trees.append(tree)
        except requests.exceptions.HTTPError:
            pass
    tree_representations = [tree.__str__() for tree in trees]
    return "\n".join(tree_representations)
