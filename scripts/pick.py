"""
Picks a number of documents from "src" and moves them to "dest"
"""

import argparse
import math
import os
import random

def main():
    parser = argparse.ArgumentParser(
        description="Picks a number of documents and places into directory."
    )
    parser.add_argument("src", type=str)
    parser.add_argument("dest", type=str)
    parser.add_argument("n", type=int, help="Number of documents to place into directory.")

    args = parser.parse_args()
    files = [os.path.join(args.src, filename) for filename in os.listdir(args.src) if filename.endswith(".txt")]
    pick(files, args.dest, args.n)

def pick(filenames, dst, n):
    random.shuffle(files)
    chosen = files[:n]

    try:
        os.mkdir(dest)
    except FileExistsError:
        pass

    for filename in chosen:
        os.rename(filename, os.path.join(dst, filename))
        print(f"Moved {filename} into {dst}...")

if __name__ == "__main__":
    main()
