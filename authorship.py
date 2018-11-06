"""
Authorship attribution classifier.
"""

import argparse
import art
import csv
import glob
import importlib
import math
import nltk
import os
import random
import shutil
import time
import yaml

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

VERBOSE = False

class Dataset():
    def __init__(self):
        self.training = []
        self.testing = []

    def contents(self):
        """Returns map of (testing, docgroup_index, doc_index) to document"""
        results = {}
        for i, docgroup in enumerate(self.training):
            content = docgroup.contents()
            for key in content:
                results[(0, i, key)] = content[key]
        for i, docgroup in enumerate(self.testing):
            content = docgroup.contents()
            for key in content:
                results[(1, i, key)] = content[key]
        return results

    def add_vectors(self, vectors):
        for (testing, docgroup_index, doc_index) in vectors:
            vector = vectors[testing, docgroup_index, doc_index]
            if testing:
                self.testing[docgroup_index].add_vector(doc_index, vector)
            else:
                self.training[docgroup_index].add_vector(doc_index, vector)

class DocumentGroup():
    def __init__(self, author, filenames, testing):
        self.author = author
        self.filenames = filenames
        self.testing = testing
        self.documents = []
        for filename in self.filenames:
            self.documents.append(Document(filename))

    def contents(self):
        return { i: self.documents[i] for i in range(len(self.documents)) }

    def add_vector(self, doc_index, vector):
        self.documents[doc_index].vector.extend(vector)

class Document():

    def __init__(self, filename):
        self.filename = filename
        self.contents = { preprocessor: open(os.path.join(filename, preprocessor)).read()
                          for preprocessor in os.listdir(self.filename) }
        self.vector = []

    def get(self, preprocessor):
        return self.contents[preprocessor]

class Runner():

    def __init__(self, config):
        self.config = config
        self.src = config["configuration"]["src"]
        self.iterations = config["configuration"]["repetitions"]
        self.authors = config["authors"]

        self.skip = config["configuration"]["skip"] if "skip" in config["configuration"] else []
        self.clean = config["configuration"]["clean"] if "clean" in config["configuration"] else []

        preprocessors = config["preprocessors"] if "preprocessors" in config else []
        self.preprocessors = [Preprocessor(p) for p in preprocessors]

        features = config["features"] if "features" in config else []
        self.features = [Feature(f) for f in features]

        self.segment_dir = os.path.join(self.src, "segments")
        self.results_dir = os.path.join(self.src, "results")
        self.results_file = os.path.join(self.results_dir, "summary.csv")

    def log(self, message, override=False):
        if override or VERBOSE:
            print(message)

    def run(self):
        """Runs all trials and gets results."""
        self.preprocess()
        for i in range(1, self.iterations + 1):
            self.run_instance(i)

    def mkdir(self, dirname):
        try:
            os.makedirs(dirname)
        except FileExistsError:
            pass

    def should_skip(self, name):
        return name in self.skip

    def preprocess(self):

        def clean():
            for dirname in self.clean:
                dirname = os.path.join(self.src, dirname)
                self.log(f"Cleaning up directory {dirname}...")
                shutil.rmtree(dirname, ignore_errors=True)

        def segment(src, dst, base, size):
            """
            Takes input files from `src`, concatenates them,
            and then segments them into files of `size` sentences
            each into the directory `dst`.
            """
            def seg(l, n):
                """Divides list l into segments of size n."""
                return [l[i:i+n] for i in range(0, len(l), n)]

            infiles = glob.glob(src)
            contents = " ".join(open(infile).read().replace("\n", " ")
                                for infile in infiles)
            sentences = nltk.sent_tokenize(contents)
            segments = seg(sentences, size)
            digits = math.ceil(math.log10(len(segments)))
            self.mkdir(dst)
            for i, segment in enumerate(segments):
                contents = " ".join(segment)
                docname = f"{base}_{str(i).zfill(digits)}"
                self.log(f"Generating segment {docname}...")
                self.mkdir(os.path.join(dst, docname))
                for preprocessor in self.preprocessors:
                    filename = os.path.join(dst, docname, f"{preprocessor.name}")
                    if os.path.exists(filename):
                        self.log(f"  Skipping {preprocessor.name} for {docname}, already exists...")
                        continue
                    self.log(f"  Using preprocessor {preprocessor.name} for {docname}...")
                    result = preprocessor.process(contents)
                    f = open(filename, "w")
                    f.write(result)
                    f.close()

        def preserve(src, dst, base):
            """Preserves the input files to the output."""
            infiles = glob.glob(src)
            digits = math.ceil(math.log10(len(infiles)))
            self.mkdir(dst)
            for i, infile in enumerate(infiles):
                contents = open(infile).read().replace("\n", " ")
                docname = f"{base}_{str(i).zfill(digits)}"
                self.log(f"Generating segment {docname}...")
                self.mkdir(os.path.join(dst, docname))
                for preprocessor in self.preprocessors:
                    self.log(f"  Using preprocessor {preprocessor.name} for {docname}...")
                    result = preprocessor.process(contents)
                    f = open(os.path.join(dst, docname, f"{preprocessor.name}"), "w")
                    f.write(result)
                    f.close()

        clean()

        # Create results file initially
        self.mkdir(self.results_dir)
        f = open(self.results_file, "w")
        f.write("iteration,accurate,inaccurate,unknown\n")
        f.close()

        if self.should_skip("preprocessing"):
            self.log("Skipping preprocessing...")
            return
        self.log("Preprocessing...")
        for segmentation in self.config["segmenters"]:
            if "preserve" in segmentation and segmentation["preserve"]:
                preserve(os.path.join(self.src, segmentation["src"]),
                         self.segment_dir,
                         self.authors[segmentation["author"]])
            else:
                segment(os.path.join(self.src, segmentation["src"]),
                        self.segment_dir,
                        self.authors[segmentation["author"]], segmentation["size"])

    def run_instance(self, i):
        """Runs a single instance of the classifier."""
        self.log(f"Running authorship classifier iteration {i}...")
        self.current_iteration = i
        self.separate()
        self.prepare_documents()
        self.train()
        self.predict()
        self.results()

    def separate(self):
        """Create the separation of testing and training files."""

        self.workdir = os.path.join(self.src, "separations", str(self.current_iteration))

        if self.should_skip("separation"):
            self.log("Skipping separation...")
            return

        # Create separations directory.
        self.log("Running separations...")
        self.mkdir(self.workdir)

        # Creat training and testing directories.
        training = os.path.join(self.workdir, "training")
        testing = os.path.join(self.workdir, "testing")
        self.mkdir(training)
        self.mkdir(testing)

        for separation in self.config["separations"]:
            self.log(f"Running separation for {separation['author']}...")
            base = self.authors[separation["author"]]
            files = glob.glob(os.path.join(self.segment_dir, f"{base}_*"))
            random.shuffle(files)
            test_count = math.floor(len(files) * separation["holdout"])
            testing_files = files[:test_count]
            training_files = files[test_count:]
            for filename in testing_files:
                basename = os.path.basename(filename)
                shutil.copytree(filename,
                                os.path.join(self.workdir, "testing", basename))
                self.log(f"  Placed {basename} into testing...")
            for filename in training_files:
                basename = os.path.basename(filename)
                shutil.copytree(filename,
                                os.path.join(self.workdir, "training", basename))
                self.log(f"  Placed {basename} into training...")

    def prepare_documents(self):
        dataset = Dataset()
        for author in self.authors:
            base = self.authors[author]
            training = glob.glob(os.path.join(self.workdir, "training", f"{base}*"))
            if training:
                group = DocumentGroup(author, training, True)
                dataset.training.append(group)
            testing = glob.glob(os.path.join(self.workdir, "testing", f"{base}*"))
            if testing:
                group = DocumentGroup(author, testing, False)
                dataset.testing.append(group)
        self.dataset = dataset

    def train(self):
        def training_labels():
            data = []
            labels = []
            for docgroup in self.dataset.training:
                for document in docgroup.documents:
                    data.append(document.vector)
                    labels.append(docgroup.author)
            return data, labels

        for feature in self.features:
            self.log(f"Generating vectors for feature {feature.config}...")
            feature.train(self.dataset)
        data, labels = training_labels()

        model = self.config["configuration"].get("model") or "NB"
        if model == "NB":
            clf = GaussianNB()
        elif model == "SVM":
            clf = SVC()
        elif model == "Perceptron":
            clf = sklearn.linear_model.Perceptron(tol=1e-3, random_state=0)
        else:
            raise Exception(f"Invalid model {model}.")
        clf.fit(data, labels)
        self.clf = clf

    def predict(self):
        for docgroup in self.dataset.testing:
            for document in docgroup.documents:
                prediction = self.clf.predict([document.vector])[0]
                document.prediction = prediction

    def results(self):
        contents = f"\n===== RESULTS {self.current_iteration} =====\n"
        accurate, inaccurate, unknown, total = 0, 0, 0, 0
        for docgroup in self.dataset.testing:
            is_unknown = docgroup.author.lower() in ["unknown", "anonymous"]
            contents += "\n" + docgroup.author + "\n"
            for document in docgroup.documents:
                total += 1
                if is_unknown:
                    unknown += 1
                elif document.prediction == docgroup.author:
                    accurate += 1
                else:
                    inaccurate += 1
                contents += f"    {document.filename}: {document.prediction}\n"

        # Summary
        accurate_percent = accurate * 100 / total
        inaccurate_percent = inaccurate * 100 / total
        unknown_percent = unknown * 100 / total
        contents += "\nSUMMARY\n"
        contents += "Accurate: {} ({:.2f}%)\n".format(accurate, accurate_percent)
        contents += "Inaccurate: {} ({:.2f}%)\n".format(inaccurate, inaccurate_percent)
        contents += "Unknown: {} ({:.2f}%)\n".format(unknown, unknown_percent)
        print(contents)

        # Log to file
        digits = math.ceil(math.log10(self.iterations))
        f = open(os.path.join(self.results_dir, f"results{str(self.current_iteration).zfill(digits)}.txt"), "w")
        f.write(contents)
        f.close()

        f = open(os.path.join(self.results_file), "a")
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow([self.current_iteration, round(accurate / total, 4), round(inaccurate / total, 4), round(unknown / total, 4)])
        f.close()

class Preprocessor():

    def __init__(self, name):
        try:
            self.name = name
            self.module = importlib.import_module(f"preprocessors.{name}")
        except ModuleNotFoundError:
            raise Exception(f"Could not find preprocessor {name}.")

    def process(self, contents):
        return self.module.preprocess(contents)

class Feature():

    def __init__(self, feature):
        try:
            self.name = feature["name"]
            self.module = importlib.import_module(f"features.{self.name}")
            self.config = feature
        except ModuleNotFoundError:
            raise Exception(f"Could not find feature {self.name}.")

    def train(self, dataset):
        contents = dataset.contents()
        vectors = self.module.train(self.config, contents)
        dataset.add_vectors(vectors)

def main():
    print(art.text2art("Authorship"))
    print(art.text2art("Attribution"))
    # time.sleep(1)
    config = parse_config()
    runner = Runner(config)
    runner.run()

def parse_config():
    global VERBOSE
    parser = argparse.ArgumentParser(
        description="Run an authorship attribution classifier on anonymous documents."
    )
    parser.add_argument("config", type=str)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    if args.verbose:
        VERBOSE = True
    contents = open(args.config).read()
    data = yaml.load(contents)
    return data

if __name__ == "__main__":
    main()
