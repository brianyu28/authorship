from types import SimpleNamespace
from sklearn.naive_bayes import GaussianNB

import argparse
import art
import glob
import importlib
import math
import nltk
import numpy as np
import os
import random
import shutil
import yaml

VERBOSE = False

class Dataset():
    """
    Maintains dataset of:
        - Training Data [0]
            - Authors
                - Sentences
        - Testing Data [1]
            - Sample Filename
                - Sentences
    """
    
    def __init__(self):
        pass
    
    def set_training(self, training):
        self.training = training

    def set_testing(self, testing):
        self.testing = testing

    def prepare(self):
        """Prepares dataset contents for passing into features and vectors."""
        self.contents = dict()
        self.vectors = dict()
        for author in self.training:
            for i, sentence in enumerate(self.training[author]):
                self.contents[(0, author, i)] = {"text": sentence}
                self.vectors[(0, author, i)] = []
        for filename in self.testing:
            for i, sentence in enumerate(self.testing[filename]):
                self.contents[(1, filename, i)] = {"text": sentence}
                self.vectors[(1, filename, i)] = []
    
    def add_vectors(self, vectors):
        for identifier in vectors:
            self.vectors[identifier].extend(vectors[identifier])

    def training_labels(self):
        """Returns data, labels for fitting"""
        data = []
        labels = []
        for author in self.training:
            for i, _ in enumerate(self.training[author]):
                data.append(self.vectors[(0, author, i)])
                labels.append(author)
        return data, labels

    def predict(self, clf):
        self.predictions = {}
        for filename in self.testing:
            vectors = []
            for i, sentence in enumerate(self.testing[filename]):
                vectors.append(self.vectors[(1, filename, i)])
            predictions = list(clf.predict(vectors))
            self.predictions[filename] = predictions
        print(self.predictions)

class Runner():

    def __init__(self, config):
        self.log("Initializing runner...")
        self.config = config

    def log(self, message, override=False):
        if override or VERBOSE:
            print(message)
    
    def run(self):
        self.clean()
        self.preprocess()
        self.load_sentences()
        self.run_instance()  # loop this line multiple times if needed
    
    def run_instance(self):
        self.log("Running instance...")
        self.training = dict()
        self.testing = dict()
        self.generate() # split into training and testing sets
        self.prepare_dataset()
        self.train()
        self.predict()

    def preprocess(self):
        self.mkdir(os.path.join(self.config.src, "sentences"))

        if self.config.should_skip("preprocessing"):
            self.log("Skipping preprocessing...")
            return

        self.log("Preprocessing...")
    
        self.log("Generating sample documents...")

        # Gather all sentences from the corpus.
        sentences = {author: [] for author in self.config.authors}
        for author_config in self.config.corpus:
            author = author_config.author
            src = glob.glob(author_config.src)
            for filename in src:
                contents = open(filename).read().replace("\n", " ")
                sentences[author].extend(nltk.sent_tokenize(contents))

        for author in self.config.authors:
            author_identifier = self.config.authors[author]
            digits = math.ceil(math.log10(len(sentences[author])))
            for i, sentence in enumerate(sentences[author]):
                dirname = f"{author_identifier}_{str(i).zfill(digits)}"
                self.log(f"Preprocessing {dirname}...")
                self.mkdir(os.path.join(self.config.src, "sentences", dirname))
                for preprocessor in self.config.preprocessors:
                    filename = os.path.join(self.config.src, "sentences", dirname, preprocessor.name)
                    if os.path.exists(filename):
                        self.log(f"    Skipping {preprocessor.name} for {dirname}, already exists...")
                        continue
                    self.log(f"    Using preprocessor {preprocessor.name} for {dirname}...")
                    result = preprocessor.process(sentence)
                    f = open(filename, "w")
                    f.write(result)
                    f.close()
    
    def load_sentences(self):
        """Loads forms of all sentences into memory, indexed using identifiers and storing a Sentence"""
        self.log("Loading sentences...")
        self.sentences = dict()
        for author in self.config.authors:
            author_identifier = self.config.authors[author]
            identifiers = glob.glob(os.path.join(self.config.src, "sentences", f"{author_identifier}_*"))
            for identifier in identifiers:
                sentence = Sentence(author, identifier)
                self.sentences[identifier] = sentence

    def generate(self):

        # Process sentences.
        sentences = dict()

        # Separate each author's sentences into testing and training set.
        for author in self.config.authors:
            author_identifier = self.config.authors[author]
            sentences = { sentence: self.sentences[sentence] for sentence in self.sentences if self.sentences[sentence].author == author }
            identifiers = list(sentences.keys())
            num_training = math.ceil(len(identifiers) * self.config.training_threshold)
            training = identifiers[:num_training]
            testing = identifiers[num_training:]
            self.training[author] = training
            self.testing[author] = testing

        # Make directory
        self.mkdir(os.path.join(self.config.src, "composite"))

        # Generate documents as a list of (author, sentence) pairs
        self.composites = []
        all_authors = list(self.config.authors.keys())
        digits = math.ceil(math.log10(self.config.generate.n))
        for i in range(self.config.generate.n):
            document = []
            authors = []
            composite = []
            author = np.random.choice(all_authors)
            while True:

                action = np.random.choice(["STAY", "NEXT", "TERMINATE"],
                          p=[self.config.generate.stay, self.config.generate.next, self.config.generate.terminate])
                if action == "TERMINATE":

                    # Document is of sufficient length, done generating.
                    if len(document) >= self.config.generate.threshold:
                        self.composites.append(composite)
                        break

                    # Document isn't long enough, try re-generating.
                    else:
                        document = []
                        authors = []
                        composite = []
                        author = np.random.choice(all_authors)
                        continue

                elif action == "NEXT":
                    author = np.random.choice([a for a in all_authors if a != author])

                # Choose a random sentence by the author.
                sentence = self.sentences[np.random.choice(self.testing[author])]
                document.append(sentence.text)
                authors.append(author)
                composite.append((author, sentence))

            doc_filename = os.path.join(self.config.src, "composite", f"{str(i).zfill(digits)}_doc.txt")
            author_filename = os.path.join(self.config.src, "composite", f"{str(i).zfill(digits)}_authors.txt")
            with open(doc_filename, "w") as f:
                for path in document:
                    f.write(path)
                    f.write("\n")
            with open(author_filename, "w") as f:
                for author in authors:
                    f.write(author)
                    f.write("\n")
            
            self.log(f"Generated document {doc_filename}...")
        
        print(self.composites)

    def prepare_dataset(self):
        self.log("Preparing dataset...")

        # Loading sentences.
        self.dataset = Dataset()
        self.dataset.set_training(self.training)
        testing_files = glob.glob(os.path.join(self.config.src, "composite", "*_doc.txt"))
        testing_data = dict()
        for testing_file in testing_files:
            testing_data[testing_file] = open(testing_file).read().splitlines()
        self.dataset.set_testing(testing_data)
        self.dataset.prepare()
    
    def train(self):
        self.log("Computing vectors...")
        for feature in self.config.features:
            vectors = feature.train(self.dataset)
        
        self.log("Fitting...")
        data, labels = self.dataset.training_labels()
        self.clf = GaussianNB()
        self.clf.fit(data, labels)

    def predict(self):
        self.dataset.predict(self.clf)

    def mkdir(self, dirname):
        try:
            os.makedirs(dirname)
        except FileExistsError:
            pass

    def clean(self):
        for dirname in self.config.clean:
            self.log(f"Cleaning up directory {dirname}...")
            shutil.rmtree(dirname, ignore_errors=True)

class Config():

    def __init__(self, contents):
        self.config = contents
        self.src = self.config["configuration"]["src"]
        self.training_threshold = self.config["configuration"]["train"]
        self.corpus = self.get_corpus()
        self.authors = self.get_authors()
        self.generate = self.get_generate_prob()
        self.features = self.get_features()
        self.preprocessors = self.get_preprocessors()
        self.clean = self.get_clean()
    
    def get_corpus(self):
        data = []
        for author_config in self.config["corpus"]:
            config = SimpleNamespace()
            config.author = author_config["author"]
            config.src = os.path.join(self.src, author_config["src"])
            data.append(config)
        return data

    def get_authors(self):
        data = {}
        for author in self.config["authors"]:
            data[author] = self.config["authors"][author]
        return data

    def get_generate_prob(self):
        data = SimpleNamespace()
        for feature in ["stay", "terminate", "threshold", "n"]:
            setattr(data, feature, self.config["generate"][feature])
        data.next = 1 - data.stay - data.terminate
        return data
    
    def get_features(self):
        data = []
        for feature in self.config["features"]:
            data.append(Feature(feature))
        return data

    def get_preprocessors(self):
        preprocessors = []
        for preprocessor in self.config["preprocessors"]:
            preprocessors.append(Preprocessor(preprocessor))
        return preprocessors

    def get_clean(self):
        return [os.path.join(self.src, dirname) for dirname in self.config["configuration"]["clean"]]
    
    def should_skip(self, step):
        return step in self.config["configuration"].get("skip", [])

class Feature():

    def __init__(self, feature):
        try:
            self.name = feature["name"]
            self.module = importlib.import_module(f"features.{self.name}")
            self.config = feature
        except ModuleNotFoundError:
            raise Exception(f"Could not find feature {self.name}.")

    def train(self, dataset):
        contents = dataset.contents
        vectors = self.module.train(self.config, contents)
        dataset.add_vectors(vectors)

class Preprocessor():

    def __init__(self, name):
        try:
            self.name = name
            self.module = importlib.import_module(f"preprocessors.{name}")
        except ModuleNotFoundError:
            raise Exception(f"Could not find preprocessor {name}.")

    def process(self, contents):
        return self.module.preprocess(contents)

class Sentence():

    def __init__(self, author, dirname):
        self.author = author
        attributes = os.listdir(dirname)
        self.attributes = { attribute: open(os.path.join(dirname, attribute)).read().rstrip("\n")
                            for attribute in attributes }

    def __getattr__(self, attr):
        if attr in self.attributes:
            return self.attributes[attr]
        else:
            raise AttributeError

def main():
    config = parse_config()
    runner = Runner(config)
    runner.run()

def parse_config():
    global VERBOSE
    parser = argparse.ArgumentParser(
        description="Run a multi-authorship attribution classifier on anonymous documents."
    )
    parser.add_argument("config", type=str)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    if args.verbose:
        VERBOSE = True
    contents = open(args.config).read()
    data = yaml.load(contents)
    return Config(data)

if __name__ == "__main__":
    print(art.text2art("Authorship"))
    print(art.text2art("Attribution"))
    main()
