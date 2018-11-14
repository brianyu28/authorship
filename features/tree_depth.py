import nltk
import numpy

from helpers import get_trees

def train(feature, texts):
    threshold = feature.get("threshold") or 5 # max counted tree depth
    vectors = {}
    for identifier in texts:
        text = texts[identifier].get("constituency")
        vectors[identifier] = get_tree_depths(text, threshold)
    return vectors

def get_tree_depths(text, threshold):
    trees = get_trees(text)
    vector = [0 for i in range(threshold)] # index n is for height n+1
    for tree in trees:
        vector[min(tree.height(), threshold) - 1] += 1
    return vector
