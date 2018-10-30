"""
replace WORD_LIST SRC DST
Replaces all occurrences of words not in WORD_LIST with DST
"""

import argparse
import nltk

def main():
    parser = argparse.ArgumentParser(
        description="Parses a source file into sample files of 20 sentences each."
    )
    parser.add_argument("words", type=str)
    parser.add_argument("src", type=str)
    parser.add_argument("dest", type=str)
    args = parser.parse_args()

    words = open(args.words).read().split()

    contents = [nltk.word_tokenize(tokens) for tokens in open(args.src).read().split()]
    filtered = [filter_words(words, tokens) for tokens in contents]
    result = " ".join(["".join(tokens) for tokens in filtered])
    f = open(args.dest, "w")
    f.write(result)
    f.close()

def filter_words(words, tokens):
    return [(word if word in words or (not any(c.isalpha() for c in word)) else "UNKNOWN")
             for word in tokens]

if __name__ == "__main__":
    main()
