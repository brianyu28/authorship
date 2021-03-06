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

import adjacency
import hmm

VERBOSE = False
ADJACENCY = True # use adjacency tracking

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
        self.run_instance()  # loop this line multiple times if needed

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
        if ADJACENCY:
            self.generate_adjacent()
        else:
            self.generate_random()
        
    def generate_random(self):
        """
        Generates documents using random sentences.
        """

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
            self.training[author] = [self.sentences[identifier] for identifier in training]
            self.testing[author] = testing

        # Make directory
        self.mkdir(os.path.join(self.config.src, "composite"))

        # Generate documents as a list of sentences pairs
        self.composites = []
        all_authors = list(self.config.authors.keys())
        digits = math.ceil(math.log10(self.config.generate.n))
        for i in range(self.config.generate.n):
            document = []
            authors = []
            composite = []
            author = random.choice(all_authors)
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
                composite.append(sentence)

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

    def generate_adjacent(self):
        """
        Generate documents consisting of adjacent sentences.
        Also trains and prepares adjacency model.
        """

        # Separate each author's sentences into sets of testing and training.
        all_segments = dict()
        self.training_segments = {author: [] for author in self.config.authors}
        for author in self.config.authors:
            self.log(f"Segmenting works of {author} into consecutive sentences...")
            author_identifier = self.config.authors[author]
            sentences = { sentence: self.sentences[sentence] for sentence in self.sentences if self.sentences[sentence].author == author }
            identifiers = list(sentences.keys())
            num_identifiers = len(identifiers)
            identifiers.sort()

            # Take the identifiers, and pull them apart into segments.
            segments = []
            (MIN_SEGMENT, MAX_SEGMENT) = (4, 10)
            while True:

                # If we have MAX_SEGMENT or fewer sentences remaining, then use all sentences as the last segment.
                if len(identifiers) <= MAX_SEGMENT:
                    segments.append(identifiers[:])
                    break

                # Create a new segment of random length.
                num_sentences = random.randint(MIN_SEGMENT, MAX_SEGMENT)
                segments.append(identifiers[:num_sentences])
                identifiers = identifiers[num_sentences:]
            random.shuffle(segments)

            # Pull training sentences until we get to the required threshold.
            training = []
            while len(training) < self.config.training_threshold * num_identifiers:
                segment = segments[0]
                self.training_segments[author].append(segment)
                segments = segments[1:]
                for sentence in segment:
                    training.append(sentence)
            self.training[author] = [self.sentences[identifier] for identifier in training]

            all_segments[author] = segments

        self.mkdir(os.path.join(self.config.src, "composite"))
        all_authors = list(self.config.authors.keys())
        self.composites = []

        # Generate n documents.
        digits = math.ceil(math.log10(self.config.generate.n))
        segments_per_document = sum(len(all_segments[author]) for author in all_authors) // self.config.generate.n
        for i in range(self.config.generate.n):

            document = []
            authors = []
            composite = []

            # Choose a random segment, add it to the document.
            for j in range(segments_per_document):
                available_authors = [author for author in all_authors if len(all_segments[author])]
                choice = random.choice(available_authors)
                segment = all_segments[choice][0]
                all_segments[choice] = all_segments[choice][1:]
                for identifier in segment:
                    sentence = self.sentences[identifier]
                    document.append(sentence.text)
                    authors.append(choice)
                    composite.append(sentence)
            
            self.composites.append(composite)

            # Write document to file.
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

        
        # Now, train adjacency model on training data.
        self.log(f"Training adjacency models...")
        self.adj_models = dict()
        for author in self.config.authors:
            training_segments = self.training_segments[author]
            adjacent = []
            nonadjacent = []

            # Keep track of used adjacent sentences, to avoid re-computing.
            used = set()

            # Take all adjacent sentences and identify score for each.
            for segment in training_segments:
                for i in range(len(segment) - 1):
                    a, b = segment[i], segment[i + 1]
                    used.add((a, b))
                    score = adjacency.score(self.sentences[a].text, self.sentences[b].text)
                    adjacent.append(score)

            # Take a sampling of non-adjacent sentences, identifying score for each.
            while len(nonadjacent) < len(adjacent):
                a = random.choice(self.training[author])
                b = random.choice(self.training[random.choice(all_authors)])
                if a == b or (a.identifier, b.identifier) in used or (b.identifier, a.identifier) in used:
                    continue
                score = adjacency.score(a.text, b.text)
                nonadjacent.append(score)

            self.adj_models[author] = adjacency.AdjacencyModel(adjacent, nonadjacent)

    def prepare_dataset(self):
        self.log("Preparing dataset...")

        # Loading sentences.
        self.dataset = Dataset()
        self.dataset.set_training(self.training)
        self.dataset.set_testing(self.composites)
        self.dataset.prepare()

    def train(self):
        self.log("Computing vectors...")
        for feature in self.config.features:
            self.log(f"Training feature {feature.name} ({feature.config})...")
            vectors = feature.train(self.dataset)

        self.log("Fitting...")
        data, labels = self.dataset.training_labels()
        self.clf = GaussianNB()
        self.clf.fit(data, labels)

    def predict(self):
        # Do the actual sentence prediction.
        self.dataset.predict(self.clf)

        # Now guess the likely sequences.
        self.dataset.composite_predictions = []
        self.dataset.composite_adj_predictions = []
        for composite in self.dataset.testing:
            model = hmm.Model(list(self.config.authors.keys()), [sentence.prediction for sentence in composite])
            model.fit()
            prediction = model.most_likely_states()
            self.dataset.composite_predictions.append(prediction)
            for i, sentence in enumerate(composite):
                sentence.composite_prediction = prediction[i]

            # If we used the adjacency model, add that prediction.
            if ADJACENCY:
                adjacencies = [None]
                for i in range(len(composite) - 1):
                    predictions = []
                    for author in self.config.authors.keys():
                        score = adjacency.score(composite[i].text, composite[i + 1].text)
                        prediction = self.adj_models[author].proba(score)
                        predictions.append(prediction)
                    adjacencies.append(predictions)
                model = hmm.Model(list(self.config.authors.keys()), [sentence.prediction for sentence in composite], adjacencies)
                model.fit()
                prediction = model.most_likely_states()
                self.dataset.composite_adj_predictions.append(prediction)
                for i, sentence in enumerate(composite):
                    sentence.composite_adj_prediction = prediction[i]

    def print_results(self):
        (accurate, inaccurate) = (0, 0)
        (composite_accurate, composite_inaccurate) = (0, 0)
        (composite_adj_accurate, composite_adj_inaccurate) = (0, 0)
        for i, composite in enumerate(self.dataset.testing):
            (c_accurate, c_inaccurate) = (0, 0)
            (c_composite_accurate, c_composite_inaccurate) = (0, 0)
            (c_composite_adj_accurate, c_composite_adj_inaccurate) = (0, 0)
            print(f"COMPOSITE {i}")
            data = []
            for j, sentence in enumerate(composite):
                if ADJACENCY:
                    data.append([sentence.identifier,
                                sentence.author,
                                sentence.prediction,
                                sentence.composite_prediction,
                                sentence.composite_adj_prediction,
                                "Yes" if sentence.author == sentence.prediction else "No",
                                "Yes" if sentence.author == sentence.composite_prediction else "No",
                                "Yes" if sentence.author == sentence.composite_adj_prediction else "No"])
                    if sentence.author == sentence.composite_adj_prediction:
                        c_composite_adj_accurate += 1
                    else:
                        c_composite_adj_inaccurate += 1
                else:
                    data.append([sentence.identifier,
                                sentence.author,
                                sentence.prediction,
                                sentence.composite_prediction,
                                "Yes" if sentence.author == sentence.prediction else "No",
                                "Yes" if sentence.author == sentence.composite_prediction else "No"])
                if sentence.author == sentence.prediction:
                    c_accurate += 1
                else:
                    c_inaccurate += 1
                if sentence.author == sentence.composite_prediction:
                    c_composite_accurate += 1
                else:
                    c_composite_inaccurate += 1
            accurate += c_accurate
            inaccurate += c_inaccurate
            composite_accurate += c_composite_accurate
            composite_inaccurate += c_composite_inaccurate
            composite_adj_accurate += c_composite_adj_accurate
            composite_adj_inaccurate += c_composite_adj_inaccurate
            if not ADJACENCY:
                print(tabulate.tabulate(data, headers=["Identifier", "Author", "Prediction", "Composite Prediction", "Accurate", "Composite Accurate"], tablefmt="psql"))
            else:
                print(tabulate.tabulate(data, headers=["Identifier", "Author", "Prediction", "Composite Pred", "Adj Pred", "Accurate", "Composite Right", "Adj Right"], tablefmt="psql"))
            accuracy = "{:.2f}%".format((c_accurate / (c_accurate + c_inaccurate)) * 100)
            composite_accuracy = "{:.2f}%".format((c_composite_accurate / (c_composite_accurate + c_composite_inaccurate)) * 100)
            print(f"Accuracy: {c_accurate} of {c_accurate + c_inaccurate} ({accuracy})")
            print(f"Composite Accuracy: {c_composite_accurate} of {c_composite_accurate + c_composite_inaccurate} ({composite_accuracy})")
            if ADJACENCY:
                composite_adj_accuracy = "{:.2f}%".format((c_composite_adj_accurate / (c_composite_adj_accurate + c_composite_adj_inaccurate)) * 100)
                print(f"Composite Adjacent Accuracy: {c_composite_adj_accurate} of {c_composite_adj_accurate + c_composite_adj_inaccurate} ({composite_adj_accuracy})")
            print()
        print()
        print("OVERALL:")
        accuracy = "{:.2f}%".format((accurate / (accurate + inaccurate)) * 100)
        composite_accuracy = "{:.2f}%".format((composite_accurate / (composite_accurate + composite_inaccurate)) * 100)
        print(f"Overall Accuracy: {accurate} of {accurate + inaccurate} ({accuracy})")
        print(f"Overall Composite Accuracy: {composite_accurate} of {composite_accurate + composite_inaccurate} ({composite_accuracy})")
        if ADJACENCY:
            composite_adj_accuracy = "{:.2f}%".format((composite_adj_accurate / (composite_adj_accurate + composite_adj_inaccurate)) * 100)
            print(f"Overall Composite Adjacent Accuracy: {composite_adj_accurate} of {composite_adj_accurate + composite_adj_inaccurate} ({composite_adj_accuracy})")
        # print(self.dataset.predictions)

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
