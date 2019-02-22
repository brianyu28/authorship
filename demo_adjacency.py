import nltk
import os
import random
import statistics

from multi import Sentence, Feature

PER_AUTHOR_COUNT = 400


def main():
    print("Adjacency Testing")

    # Open files.
    filenames = os.listdir("data/bronte/sentences")
    authors = ["anne", "emily", "charlotte"]
    filenames_by_author = {author: [] for author in authors}

    for filename in filenames:
        for author in authors:
            if filename.startswith(author):
                filenames_by_author[author].append(filename)
    for author in authors:
        filenames_by_author[author].sort()
        filenames_by_author[author] = filenames_by_author[author][100:100 + PER_AUTHOR_COUNT]

    print("Loading sentences...")
    sentences = {author: [] for author in authors}
    for author in authors:
        print("Loading sentences for author", author, "...")
        for filename in filenames_by_author[author]:
            s = Sentence(author, os.path.join("data/bronte/sentences/", filename))
            sentences[author].append(s)
    
    print("Sentences loaded.")
    valid_pairs = []
    invalid_pairs = []
    off_by_two_pairs = []
    for author in authors:
        for _ in range(50):
            i = random.randint(0, PER_AUTHOR_COUNT - 2)
            valid_pairs.append((sentences[author][i], sentences[author][i + 1]))
        for _ in range(50):
            i = random.randint(0, PER_AUTHOR_COUNT - 3)
            off_by_two_pairs.append((sentences[author][i], sentences[author][i + 2]))
        for _ in range(50):
            i = random.randint(0, PER_AUTHOR_COUNT - 1)
            j = random.randint(0, PER_AUTHOR_COUNT - 1)
            if abs(i - j) > 1:
                invalid_pairs.append((sentences[author][i], sentences[author][j]))

    data_on_pairs("VALID PAIRS", numbers_for_pairs(valid_pairs))
    data_on_pairs("OFF BY TWO PAIRS", numbers_for_pairs(off_by_two_pairs))
    data_on_pairs("INVALID PAIRS", numbers_for_pairs(invalid_pairs))


def numbers_for_pairs(pairs):
    numbers = []
    for s1, s2 in pairs:
        numbers.append(word_similarity(s1.text, s2.text))
    return numbers


def word_similarity(text1, text2):
    tokens1 = set(token.lower() for token in nltk.word_tokenize(text1) if any(c.isalpha() for c in token))
    tokens2 = set(token.lower() for token in nltk.word_tokenize(text2) if any (c.isalpha() for c in token))
    # print(tokens1.intersection(tokens2))
    return len(tokens1.intersection(tokens2)) / len(tokens1.union(tokens2))


def data_on_pairs(name, lst):
    print()
    print(name)
    print("Average Similarity: {0:.4f}%".format(statistics.mean(lst) * 100))
    print("Median Similarity:  {0:.4f}%".format(statistics.median(lst) * 100))
    print("St Dev Similarity:  {0:.4f}%".format(statistics.stdev(lst) * 100))
    print("N:  {}".format(len(lst)))


if __name__ == "__main__":
    main()