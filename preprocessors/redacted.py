import nltk

def preprocess(contents):
    """This is the identity function: return the original text unchanged."""
    words = open("resources/common_words.txt").read().split()
    contents = [nltk.word_tokenize(tokens) for tokens in contents.split()]
    filtered = [filter_words(words, tokens) for tokens in contents]
    result = " ".join(["".join(tokens) for tokens in filtered])
    return result

def filter_words(words, tokens):
    return [(word if word.lower() in words or (not any(c.isalpha() for c in word)) else "UNKNOWN")
             for word in tokens]
