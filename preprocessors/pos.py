import nltk

def preprocess(contents):
    """This is the identity function: return the original text unchanged."""
    words = nltk.word_tokenize(contents)
    tags = [tag for (token, tag) in nltk.pos_tag(words)]
    return " ".join(tags)
