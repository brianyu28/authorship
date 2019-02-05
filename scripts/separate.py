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
        description="Splits documents into a number of different sets."
    )
    parser.add_argument("dirname", type=str)
    parser.add_argument("number", type=int)

    args = parser.parse_args()
    files = os.listdir(args.dirname)

    # Create all of the result directories.
    for i in range(args.number):
        try:
            print("Creating directory", (args.dirname + str(i)), "...")
            os.mkdir(args.dirname + str(i))
        except FileExistsError:
            pass

    for filename in files:
        choice = random.choice(list(range(args.number)))
        print("Sorting file", filename, "into choice", choice, "...")
        os.rename(os.path.join(args.dirname, filename),
                  os.path.join(args.dirname + str(choice), filename))

if __name__ == "__main__":
    main()
