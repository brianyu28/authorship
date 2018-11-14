import nltk

from itertools import dropwhile

def prune_counter(counter, threshold):
    for key, count in dropwhile(lambda key_count: key_count[1] >= threshold, counter.most_common()):
        del counter[key]

def is_training(identifier):
    return identifier[0] == 0

def process_trees(text):
    """Takes a text representation of trees and returns the trees."""
    trees = []
    return trees

def replace_smart(text):
    return text.replace("‘", "'").replace("’", "'").replace("“", "\"").replace("”", "\"")

def get_trees(contents):
    """Gets a list of NLTK trees given an input of trees."""
    root = "(ROOT"
    text_trees = [root + tree for tree in contents.split(root) if tree]
    trees = []
    for tree in text_trees:
        try:
            trees.append(nltk.tree.Tree.fromstring(tree))
        except ValueError:
            pass
    return trees

