"""
Concatenates a number of files into a single file.
"""
import glob

def main():
    source = input("Input glob: ")
    filenames = list(glob.glob(source))
    output = input("Output file: ")
    concat(filenames, output)

def concat(inputs, output):
    """Concatenates inputs into output."""
    with open(output, "w") as outfile:
        for fname in inputs:
            with open(fname, errors="ignore") as infile:
                outfile.write(infile.read())

if __name__ == "__main__":
    main()
