"""
lopsn-gram
Lexical Order Preserving Syntacitc n-gram
n-grams that get at syntactic structure while preserving the lexical order of words.
For n = 1, 2; this is the same as regular n-grams

Run from inside of a data directory
"""

import os
from nltk.tree import Tree

redacted = False
N = 5
pos = True
words = open("../../resources/common_words.txt").read().split()

def extract_lops_from_dir(dirname):
    directories = os.listdir(dirname)
    for i, directory in enumerate(directories):
        print(f"Processing {i + 1} of {len(directories)} sentences: {dirname}/{directory}...")
        lops = extract_lops_from_filename(os.path.join(dirname, directory, "constituency"), N)
        with open(os.path.join(dirname, directory, f"lops{N}{'_pos' if pos else ''}{'_redacted' if redacted else ''}"), "w") as f:
            f.write("\n".join(lops))

def extract_lops_from_filename(filename, n):
    f = open(filename)
    t = Tree.fromstring(f.read())
    lops = extract_lops(t, n)
    return lops

def redact(word):
    if redacted:
        return word if word.lower() in words or (not any(c.isalpha() for c in word)) else "UNKNOWN"
    else:
        return word

def extract_lops(tree, n):
    paths = []
    def extract_paths(tree, path):
        if pos and len(tree) == 1 and isinstance(tree[0], str):
            paths.append((path, tree.label()))
        elif isinstance(tree, str):
            paths.append((path, redact(tree)))
        elif isinstance(tree, Tree):
            for i, child in enumerate(tree):
                extract_paths(child, path + [i])
    extract_paths(tree, [])
    lops = []
    for i in range(len(paths) - n + 1):
        seq = [(path.copy(), word) for (path, word) in paths[i:i+n]]
        lops.append(extract_lop(seq))
    return lops

def extract_lop(seq):
    if len(seq) == 1:
        return "(" + seq[0][1] + ")"
    branches = {}
    for path in seq:
        direction = path[0][0]
        if direction not in branches:
            branches[direction] = []
        del path[0][0]
        branches[direction].append(path)
    if len(branches) == 1:
        return extract_lop(seq)
    children = [extract_lop(branches[d]) for d in branches]
    return "(" + "".join(children) + ")"

if __name__ == "__main__":
    for i in range(10):
        extract_lops_from_dir(f"sentences{i}")
