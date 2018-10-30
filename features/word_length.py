import numpy

def train(feature, texts):
    vectors = {}
    for identifier in texts:
        text = texts[identifier].get("text")
        vectors[identifier] = [average_word_length(text)]
    return vectors

def average_word_length(text):
    words = text.split(" ")
    return numpy.mean([len(word) for word in words])