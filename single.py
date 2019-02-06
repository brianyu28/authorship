"""
Variant on Multi-Author Attribution System for Single-Author Attribution

Two evaluation methods:
- whole document vector
- sentence-level vectors and taking the plurality
"""

from collections import Counter
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
import sys
import tabulate
import yaml

import markov

# (a, b, c) correspond tot hese values
TRAINING_DOCUMENTS_PER_AUTHOR = 10 # Train = n
TESTING_DOCUMENTS_PER_AUTHOR = 50
SENTENCES_PER_DOCUMENT = 50

VERBOSE = False

class Dataset():
    """

    Two instantiations.
    Whole Document Vector
    Training [0]: (0, author, sentence_index)
    Testing [1]: (1, document_index, 0) <- 0 because there will only ever be one sentence index

    Sentence-Level Vectors
    Training [0]: (0, author, sentence_index)
    Testing [1]: (1, document_index, sentence_index)
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
                self.contents[(0, author, i)] = sentence
                self.vectors[(0, author, i)] = []
        for i, composite in enumerate(self.testing):
            for j, sentence in enumerate(composite):
                self.contents[(1, i, j)] = sentence
                self.vectors[(1, i, j)] = []

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
        self.predictions = []
        for i, composite in enumerate(self.testing):
            vectors = []
            for j, sentence in enumerate(composite):
                vectors.append(self.vectors[(1, i, j)])
            predictions = list(clf.predict(vectors))
            self.predictions.append(predictions)
            for j, sentence in enumerate(composite):
                sentence.prediction = predictions[j]

class Runner():

    def __init__(self, config):
        self.log("Initializing runner...")
        self.config = config

    def log(self, message, override=False, clearline=False):
        if override or VERBOSE:
            if clearline:
                sys.stdout.write("\033[F")
            print(message)

    def run(self):
        self.clean()
        self.preprocess()
        self.load_sentences()
        for i in range(10):
            print(f"START INSTANCE {i}")
            self.run_instance()  # loop this line multiple times if needed
            print(f"END INSTANCE {i}")

    def run_instance(self):
        self.log("Running instance...")
        self.training = dict()
        self.testing = dict()
        self.generate() # split into training and testing sets
        self.prepare_dataset()
        self.train()
        self.predict()
        self.print_results()

    def preprocess(self):
        self.mkdir(os.path.join(self.config.src, self.config.sentence_dir))

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
                self.mkdir(os.path.join(self.config.src, self.config.sentence_dir, dirname))
                for preprocessor in self.config.preprocessors:
                    filename = os.path.join(self.config.src, self.config.sentence_dir, dirname, preprocessor.name)
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
        for i, author in enumerate(self.config.authors):
            author_identifier = self.config.authors[author]
            identifiers = glob.glob(os.path.join(self.config.src, self.config.sentence_dir, f"{author_identifier}_*"))
            for j, identifier in enumerate(identifiers):
                self.log(f"    Loading sentence {j + 1} of {len(identifiers)} for author {i+1} of {len(self.config.authors)}...", clearline=i > 0 or j > 0)
                sentence = Sentence(author, identifier)
                self.sentences[identifier] = sentence

    def generate(self):
        
        def generate_document_from_sentences(sentences, n):
            sampled = [] 
            for i in range(n):
                k = random.randint(0, len(sentences) - 1)
                sampled.append(sentences[k])
                del sentences[k]
            return sampled
        
        self.training_documents = {}
        self.testing_documents = []

        for author in self.config.authors:
            self.training_documents[author] = []
            sentences = { sentence: self.sentences[sentence] for sentence in self.sentences if self.sentences[sentence].author == author }
            identifiers = list(sentences.keys())

            for i in range(TRAINING_DOCUMENTS_PER_AUTHOR):
                sampled = generate_document_from_sentences(identifiers, SENTENCES_PER_DOCUMENT)
                sentences = [self.sentences[identifier] for identifier in sampled]
                document = Document(author, f"Training Document {i}", sentences)
                self.training_documents[author].append(document)

            for i in range(TESTING_DOCUMENTS_PER_AUTHOR):
                sampled = generate_document_from_sentences(identifiers, SENTENCES_PER_DOCUMENT)
                sentences = [self.sentences[identifier] for identifier in sampled]
                document = Document(author, f"Testing Document {i}", sentences)
                self.testing_documents.append(document)

        self.log(f"Done generating documents.")

    def prepare_dataset(self):
        self.log("Preparing datasets...")

        # Loading sentences into two datasets, one for the whole dataset and one for sentences.
        self.whole_dataset = Dataset()
        self.whole_dataset.set_training(self.training_documents)
        self.whole_dataset.set_testing([[document] for document in self.testing_documents])
        self.whole_dataset.prepare()

        self.sentence_dataset = Dataset()
        training = {}
        for author in self.training_documents:
            training[author] = []
            for document in self.training_documents[author]:
                training[author].extend(document.sentences)
        testing = []
        for document in self.testing_documents:
            testing.append(document.sentences)
        self.sentence_dataset.set_training(training)
        self.sentence_dataset.set_testing(testing)
        self.sentence_dataset.prepare()
        print("Done preparing dataset.")

    def train(self):
        self.log("Computing vectors...")
        for feature in self.config.features:
            self.log(f"Training feature {feature.name} ({feature.config})...")
            vectors = feature.train(self.whole_dataset)
            vectors = feature.train(self.sentence_dataset)

        self.log("Fitting...")
        data, labels = self.whole_dataset.training_labels()
        self.whole_clf = GaussianNB()
        self.whole_clf.fit(data, labels)

        data, labels = self.sentence_dataset.training_labels()
        self.sentence_clf = GaussianNB()
        self.sentence_clf.fit(data, labels)

    def predict(self):
        # Do the actual sentence prediction.
        self.log("Predicting...")
        self.whole_dataset.predict(self.whole_clf)
        self.sentence_dataset.predict(self.sentence_clf)

    def print_results(self):
        (overall_accurate, overall_inaccurate) = (0, 0)
        (sentence_accurate, sentence_inaccurate) = (0, 0)

        data = []
        for i in range(len(self.whole_dataset.testing)):
            author = self.whole_dataset.testing[i][0].author
            whole_prediction = self.whole_dataset.testing[i][0].prediction
            sentence_counts = Counter([sentence.prediction for sentence in self.sentence_dataset.testing[i]])
            sentence_prediction = sentence_counts.most_common(1)[0][0]
            data.append([i, author, whole_prediction, sentence_prediction, sentence_counts])
            if author == whole_prediction:
                overall_accurate += 1
            else:
                overall_inaccurate += 1

            if author == sentence_prediction:
                sentence_accurate += 1
            else:
                sentence_inaccurate += 1

        print(tabulate.tabulate(data, headers=["Identifier", "Author", "Whole Prediction", "Sentence Prediction", "Counts"], tablefmt="psql"))
        overall_accuracy = "{:.2f}%".format((overall_accurate / (overall_accurate + overall_inaccurate)) * 100)
        sentence_accuracy = "{:.2f}%".format((sentence_accurate / (sentence_accurate + sentence_inaccurate)) * 100)
        print(f"Overall Accuracy: {overall_accurate} of {overall_accurate + overall_inaccurate} ({overall_accuracy})")
        print(f"Sentence Accuracy: {sentence_accurate} of {sentence_accurate + sentence_inaccurate} ({sentence_accuracy})")

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
        self.sentence_dir = self.config["configuration"]["sentence_dir"]
        self.training_threshold = self.config["configuration"]["train"]
        self.accuracy = self.config["configuration"]["accuracy"]
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
        self.identifier = dirname
        attributes = os.listdir(dirname)
        self.attributes = { attribute: open(os.path.join(dirname, attribute)).read().rstrip("\n")
                            for attribute in attributes }

    def __getattr__(self, attr):
        if attr in self.attributes:
            return self.attributes[attr]
        else:
            print(attr)
            print(self.attributes)
            print(attr in self.attributes)
            raise AttributeError

    def get(self, attr):
        return self.__getattr__(attr)

class Document(Sentence):
    """
    A document is just a list of sentences.
    """
    
    def __init__(self, author, identifier, sentences):
        self.author = author
        self.identifier = identifier
        self.sentences = sentences

        # Combine sentences attributes.
        self.attributes = {}
        for sentence in self.sentences:
            for attr in sentence.attributes:
                if attr not in self.attributes:
                    self.attributes[attr] = sentence.attributes[attr]
                else:
                    self.attributes[attr] += "\n"
                    self.attributes[attr] += sentence.attributes[attr]


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
