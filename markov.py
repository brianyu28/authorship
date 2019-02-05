"""
Library to predict authorship assignments from a document.
"""
import random
from pomegranate import *

def predict_assignments(authors, assignments, accuracy):
    """
    Predictions underlying assignments for `authors`
    Given a list of emission assignments `assignments`
    Given that the emissions are accurate with probability `accuracy`
    """
    distributions = [distribution_for_author(author, authors, accuracy) for author in authors]
    stay = 0.25 + (random.random() / 2) # generate random number between .25 and .75
    transition_matrix = [[stay if a == author else (1-stay)/(len(authors) - 1) for a in authors] for author in authors]
    starts = [1/len(authors)] * len(authors)
    hmm = HiddenMarkovModel.from_matrix(transition_matrix, distributions, starts, state_names=authors)
    hmm.fit([assignments]) # train data using Baum-Welch
    prediction = hmm.predict(assignments)
    states = [state.name for state in hmm.states]
    return [states[i] for i in prediction]

def distribution_for_author(author, authors, accuracy):
    """
    Returns a discrete distribution for the `author`
    in the list of all `authors` that is accurate with probability `accuracy`.
    """
    self_proba = accuracy if len(authors) > 1 else 1
    other_proba = (1 - accuracy) / (len(authors) - 1) if len(authors) > 1 else 0
    return DiscreteDistribution({ a : (self_proba if a == author else other_proba) for a in authors })

