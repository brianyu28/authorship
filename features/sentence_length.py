import nltk
import numpy

def train(feature, texts):
    vectors = {}
    for identifier in texts:
        text = texts[identifier].get("text")
        vectors[identifier] = [average_sentence_length(text)]
    return vectors

def average_sentence_length(text):
    sentences = nltk.sent_tokenize(text)
    return numpy.mean([len(sentence) for sentence in sentences])