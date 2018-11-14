import nltk
import numpy

from helpers import get_trees

def train(feature, texts):
    threshold = feature.get("threshold") or 5 # max counted tree constituent end
    vectors = {}
    for identifier in texts:
        text = texts[identifier].get("constituency")
        vectors[identifier] = get_tree_constituent_ends(text, threshold)
    return vectors

def get_tree_constituent_ends(text, threshold):
    trees = get_trees(text)
    vector = [0 for i in range(threshold)] # index n is for number of constituents n+1
    def compute(tree, endings):
        if len(tree) == 1:  # if tree is a leaf node
            vector[min(endings + 1, threshold) - 1] += 1
        else:
            for i, subtree in enumerate(tree):
                print(subtree)
                if i == len(tree) - 1:  # last one
                    compute(subtree, endings + 1)
                else:
                    compute(subtree, endings)
    for tree in trees:
        compute(tree, 0)
    return vector

