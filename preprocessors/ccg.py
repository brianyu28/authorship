import nltk
import requests
from nltk.parse import CoreNLPParser

from helpers import replace_smart

SERVER = "http://127.0.0.1:5000/ccg"

def preprocess(contents):
    """For preprocessing a single sentence."""
    tags = requests.post(SERVER, data={"data": contents}, timeout=10)
    if tags.status_code != 200:
        return "NONE"
    else:
        return tags.text


def preprocess_many(contents):
    """For preprocessing many sentences."""
    sentences = nltk.sent_tokenize(replace_smart(contents))
    all_tags = []
    for sentence in sentences:
        try:
            tags = requests.post(SERVER, data={"data": sentence}, timeout=10)
            if tags.status_code != 200:
                continue
            all_tags.append(tags.text)
        except (requests.exceptions.HTTPError, requests.exceptions.Timeout):
            continue
    return "\n".join(all_tags)
