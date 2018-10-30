"""
Generates documents consisting of random sentences from a source document.
"""

import argparse
import math
import nltk
import os
import random
import sys

def main():

    # Parse arguments.
    parser = argparse.ArgumentParser(
        description="Parses a source file into sample files of 20 sentences each."
    )
    parser.add_argument("filename", type=str)
    parser.add_argument("-n", "--num", type=int, default=100,
        help="Number of documents.")
    parser.add_argument("-s", "--size", type=int, default=20,
        help="Document size.")
    parser.add_argument("-d", "--directory", default="out",
        help="Output directory name.")
    parser.add_argument("-b", "--basename", default="text",
        help="Basename for output text.")
    args = parser.parse_args()
    contents = open(args.filename).read().replace("\n", " ")
    sentences = nltk.sent_tokenize(contents)
    print("Processed sentences.")

    # Create new documents from input file
    digits = math.ceil(math.log10(args.num))
    try:
        os.mkdir(args.directory)
    except FileExistsError:
        pass
    for i in range(args.num):
        chosen = sentences.copy()
        random.shuffle(chosen)
        chosen = chosen[:args.size]
        contents = " ".join(chosen)
        filename = f"{args.basename}{str(i).zfill(digits)}.txt"
        f = open(os.path.join(args.directory, filename), "w")
        f.write(contents)
        f.close()
        print(f"Wrote file {filename}...")

if __name__ == "__main__":
    main()
