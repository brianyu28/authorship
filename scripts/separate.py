"""
Separates the contents of a directory into training and testing documents.

Takes as arguments:
    - directory name with files to separate
    - (-p) proportion of documents that should be in testing set (default 0.2)
"""

import argparse
import math
import os
import random

def main():
    parser = argparse.ArgumentParser(
        description="Splits documents into training and testing set."
    )
    parser.add_argument("dirname", type=str)
    parser.add_argument("-p", "--proportion", type=float, default=0.2,
        help="Proportion of documents to put into testing.")

    args = parser.parse_args()
    files = os.listdir(args.dirname)
    random.shuffle(files)
    test_count = math.floor(len(files) * args.proportion)
    testing = files[:test_count]
    training = files[test_count:]

    try:
        os.mkdir(os.path.join(args.dirname, "testing"))
    except FileExistsError:
        pass
    try:
        os.mkdir(os.path.join(args.dirname, "training"))
    except FileExistsError:
        pass

    for filename in testing:
        os.rename(os.path.join(args.dirname, filename),
                  os.path.join(args.dirname, "testing", filename))
        print(f"Sorted {filename} into testing...")
    for filename in training:
        os.rename(os.path.join(args.dirname, filename),
                  os.path.join(args.dirname, "training", filename))
        print(f"Sorted {filename} into training...")

if __name__ == "__main__":
    main()
