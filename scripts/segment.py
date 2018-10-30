"""
Segments a document into files of equal number of sentences.
"""

import argparse
import math
import nltk
import os

def main():

    # Parse arguments.
    parser = argparse.ArgumentParser(
        description="Splits text into separate documents of some sentence length."
    )
    parser.add_argument("filename", type=str)
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

    segments = seg(sentences, args.size)
    digits = math.ceil(math.log10(len(segments)))

    try:
        os.mkdir(args.directory)
    except FileExistsError:
        pass
    for i, segment in enumerate(segments):
        contents = " ".join(segment)
        filename = f"{args.basename}{str(i).zfill(digits)}.txt"
        f = open(os.path.join(args.directory, filename), "w")
        f.write(contents)
        f.close()
        print(f"Wrote file {filename}...")

# Divides list l into segments of size n
def seg(l, n):
    return [l[i:i+n] for i in range(0, len(l), n)]

if __name__ == "__main__":
    main()
