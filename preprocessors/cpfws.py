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
    tree = cpfws(tree)
    subtrees = all_subtrees(tree)
    return "\n".join([cpfws2string(subtree) for subtree in subtrees])

function_words = open("resources/function_words.txt").read().splitlines()

def cpfws(tree):
    # if it's a function word, then it's just that
    if isinstance(tree, str):
        if tree.lower() in function_words:
            return [tree.lower()]
        else:
            return []
    # if it's a tree, then recursively search children
    elif isinstance(tree, nltk.Tree):
        children = list(filter(lambda child: len(child) > 0, [cpfws(child) for child in tree]))
        if len(children) == 1:
            return children[0]
        else:
            return children
    # not a string or a tree
    else:
        return []

def all_subtrees(tree):
    if isinstance(tree, str):
        return []
    results = [tree]
    for subtree in tree:
        results.extend(all_subtrees(subtree))
    return results

def cpfws2string(subtree):
    # string is just the string
    if isinstance(subtree, str):
        return subtree
    result = "("
    for child in subtree:
        result += cpfws2string(child)
    result += ")"
    return result
