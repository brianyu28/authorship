"""
Implementation of Baum Welch.
Allows for a modification of Baum-Welch that breaks some of the assumptions
of the Hidden Markov Model, by using a separate model to predict whether two
sentences are adjacent to one another.
"""

import numpy as np
import random

class Model:
    """
    Hidden Markov Model for authorship attribution.

    INPUTS:
    X[T] : X[i] = hidden states (actual author) at time i
    Y[T] : Y[i] = observed state (predicted author) at time i
    Z[T] : Z[i] = probability that sentence i is by the same author as i-1
        ^-- new addition to model, Z[0] = None

    MODEL:
    A[N][N] : A[i][j] = P(X_t = j | X_{t-1} = i) <-- transitions in actual authors
    Pi[N] : Pi[i] = P(X_1 = i) <-- initial state distribution of actual authors
    B[N][N] : B[i][j] = P(Y_t = j | X_t = i) <-- emission distribution
    Describe Markov Chain by Theta=(A, B, Pi)

    FORWARD-BACKWARD COMPUTATION:
    Alpha[N][T] = Alpha[i][t] = P(Y_1 = y_1, ..., Y_t = y_t, X_t = i | Theta)
        ^-- probability of seeing y_1, ..., y_t and being in state i at time t

    Beta[N][T] = Beta[i][t] = P(Y_{t+1} = y_{t+1}, ..., Y_T = y_T | X_t = i, Theta)
        ^-- probability of ending sequence y_{t+1}, ..., y_T given state i at time t

    UPDATE VARIABLES:
    Gamma[N][T] = Gamma[i][t] = P(X_t = i | Y, Theta)
        ^-- probability of being in state i at time t, given observations and model

    Xi[N][N][T-1] = Xi[i][j][t] = P(X_t = 1, X_{t+1} = j | Y, Theta)
        ^-- probability of being in states i and j at times t and t+1, given Y, model
    """

    def __init__(self, authors, predictions, adjacencies=None):
        self.N = len(authors)
        self.T = len(predictions)
        self.authors = authors

        # Maintain a mapping of authors to their indices.
        self.author_idx = {}
        for i, author in enumerate(authors):
            self.author_idx[author] = i

        self.Y = predictions
        self.use_adjacencies = adjacencies is not None
        self.adjacencies = adjacencies

    def fit(self):
        """
        Runs the Baum-Welch algorithm to estimate the parameters of the model.
        """
        N, T = self.N, self.T

        # Generate a random initial set of parameters.
        parameters = Parameters(N)
        parameters.randomize()

        # Loop until convergence.
        iterations = 0
        while True:
            if iterations % 100 == 0:
                print(f"Baum-Welch estimation, iteration {iterations}...")
            iterations += 1
            Alpha, Beta = self.forward_backward(parameters, self.Y, self.adjacencies)
            new_parameters = self.update(parameters, self.Y, Alpha, Beta)
            distance = parameters.distance_from(new_parameters)
            if distance < 2 ** -16:
                break
            parameters = new_parameters
        self.parameters = parameters

    def forward_backward(self, parameters, predictions, adjacencies):
        """
        Alpha[N][T] = Alpha[i][t] = P(Y_1 = y_1, ..., Y_t = y_t, X_t = i | Theta)
            ^-- probability of seeing y_1, ..., y_t and being in state i at time t

        Beta[N][T] = Beta[i][t] = P(Y_{t+1} = y_{t+1}, ..., Y_T = y_T | X_t = i, Theta)
            ^-- probability of ending sequence y_{t+1}, ..., y_T given state i at time t
        """
        N = parameters.N
        T = len(predictions)
        Y = predictions
        A, Pi, B = parameters.A, parameters.Pi, parameters.B
        author = self.author_idx

        # Initialize alpha and beta to empty matrices.
        Alpha = [[None for _ in range(T)] for _ in range(N)]
        Beta = [[None for _ in range(T)] for _ in range(N)]

        # Compute Alpha using the forward procedure:
        # Probability of seeing observations through t, and being in state i.

        # Compute Alpha[i][0], base case for when t = 0
        for i in range(N):
            Alpha[i][0] = Pi[i] * B[i][author[Y[0]]]

        # Recursively compute A[i][t] for t = 1, ..., T-1
        for t in range(1, T):
            for i in range(N):

                # Sum over probabilities of all possible ways to get to state i.
                if not self.use_adjacencies:
                    prob_state = sum(Alpha[j][t-1] * A[j][i] for j in range(N))
                else:
                    prob_state = sum((Alpha[j][t-1] * A[j][i])
                                     if j != i else
                                     (Alpha[j][t-1] * adjacencies[t])
                                     for j in range(N))

                # Probability of emission of observed state.
                prob_emission = B[i][author[Y[t]]]

                Alpha[i][t] = prob_state * prob_emission

        # Compute Beta using the backward procedure:
        # Probability of seeing ending sequence given starting in state i at time t.

        # Compute Beta[i][T-1], base case for when t = T - 1
        for i in range(N):
            Beta[i][T - 1] = 1

        # Recursively compute B[i][t] for t = T-2, ..., 0
        for t in range(T - 2, -1, -1):
            for i in range(N):
                if not self.use_adjacencies:
                    Beta[i][t] = sum(Beta[j][t + 1] * A[i][j] * B[j][author[Y[t+1]]]
                                    for j in range(N))
                else:
                    Beta[i][t] = sum((Beta[j][t + 1] * A[i][j] * B[j][author[Y[t+1]]])
                                    if j != i else
                                    (Beta[j][t + 1] * adjacencies[t+1] * B[j][author[Y[t+1]]])
                                    for j in range(N))
        return Alpha, Beta

    def update(self, parameters, predictions, Alpha, Beta):
        """
        Gamma[N][T] = Gamma[i][t] = P(X_t = i | Y, Theta)
            ^-- probability of being in state i at time t, given observations and model

        Xi[N][N][T-1] = Xi[i][j][t] = P(X_t = 1, X_{t+1} = j | Y, Theta)
            ^-- probability of being in states i and j at times t and t+1, given Y, model
        """
        N = parameters.N
        T = len(predictions)
        Y = predictions
        A, Pi, B = parameters.A, parameters.Pi, parameters.B
        author = self.author_idx

        # Compuet Gamma[i][t], probability of being in state i at time t
        Gamma = [[None for _ in range(T)] for _ in range(N)]
        for t in range(T):
            non_normalized = [Alpha[i][t] * Beta[i][t] for i in range(N)]
            normalized = [x / sum(non_normalized) for x in non_normalized]
            for i, prob in enumerate(normalized):
                Gamma[i][t] = prob

        # Compute Xi[i][j][t], probability of being in states i and j at times t and t+1
        Xi = [[[None for _ in range(T - 1)] for _ in range(N)] for _ in range(N)]
        for t in range(T - 1):
            non_normalized = [[Alpha[i][t] * A[i][j] * Beta[j][t + 1] * B[j][author[Y[t+1]]]
                              for j in range(N)] for i in range(N)] # index [i][j]
            total = sum(sum(row) for row in non_normalized)
            normalized = [[x / total for x in row] for row in non_normalized]
            for i, row in enumerate(normalized):
                for j, prob in enumerate(row):
                    Xi[i][j][t] = prob

        # Generate new parameters.
        p = Parameters(N)

        # Compute new Pi.
        for i in range(N):
            p.Pi[i] = Gamma[i][0]

        # Compute new A.
        for i in range(N):
            denominator = sum(Gamma[i][t] for t in range(T - 1))
            for j in range(N):
                numerator = sum(Xi[i][j][t] for t in range(T - 1))
                p.A[i][j] = numerator / denominator

        # Compute new B.
        for i in range(N):
            denominator = sum(Gamma[i][t] for t in range(T))
            for j in range(N):
                numerator = sum(Gamma[i][t] if author[Y[t]] == j else 0 for t in range(T))
                p.B[i][j] = numerator / denominator
        return p

    def most_likely_states(self):
        """
        Computes the most likely states at each point in time by computing Gamma.
        """
        N = self.parameters.N
        Y = self.Y
        T = len(Y)
        A, Pi, B = self.parameters.A, self.parameters.Pi, self.parameters.B
        author = self.author_idx

        Alpha, Beta = self.forward_backward(self.parameters, Y, self.adjacencies)

        # Compuet Gamma[i][t], probability of being in state i at time t
        Gamma = [[None for _ in range(T)] for _ in range(N)]
        for t in range(T):
            non_normalized = [Alpha[i][t] * Beta[i][t] for i in range(N)]
            normalized = [x / sum(non_normalized) for x in non_normalized]
            for i, prob in enumerate(normalized):
                Gamma[i][t] = prob

        states = [self.authors[np.argmax([Gamma[i][t] for i in range(N)])] for t in range(T)]
        return states


class Parameters:
    """
    Parameters for the HMM.
    Stores values:
        N := number of authors

        A[N][N] : A[i][j] = P(X_t = j | X_{t-1} = i) <-- transitions in actual authors
        Pi[N] : Pi[i] = P(X_1 = i) <-- initial state distribution of actual authors
        B[N][N] : B[i][j] = P(Y_t = j | X_t = i) <-- emission distribution
    """

    def __init__(self, n):
        self.N = n
        self.A = [[None for _ in range(self.N)]for _ in range(self.N)]
        self.B = [[None for _ in range(self.N)]for _ in range(self.N)]
        self.Pi = [None for _ in range(self.N)]

    def randomize(self):
        N = self.N

        # Randomly generate transition probabilities.
        for i in range(N):
            raw = [0.25 + (random.random() / 2) for _ in range(N)]
            normalized = [x / sum(raw) for x in raw]
            for j in range(N):
                self.A[i][j] = normalized[j]

        # Generate emission probabilities using heuristic, assume about 0.6 right.
        for i in range(N):
            raw = [2 if i == k else 1 for k in range(N)]
            normalized = [x / sum(raw) for x in raw]
            for j in range(N):
                self.B[i][j] = normalized[j]

        # Randomly generate initial state distribution.
        raw = [random.random() for _ in range(N)]
        normalized = [x / sum(raw) for x in raw]
        for i in range(N):
            self.Pi[i] = normalized[i]

    def distance_from(self, other):
        """
        Computes average parameter difference between this parameter set and another.
        """
        N = self.N
        total = 2 * N * N + N
        difference = 0
        for i in range(N):
            difference += abs(self.Pi[i] - other.Pi[i])
            for j in range(N):
                difference += abs(self.A[i][j] - other.A[i][j])
                difference += abs(self.B[i][j] - other.B[i][j])
        return difference / total

    def print(self):
        N = self.N
        A, B, Pi = self.A, self.B, self.Pi
        print("PARAMETERS")
        print("=========")
        print("Initial State Distribution")
        for i in range(N):
            print("  Author {}: {:.4f}".format(i, Pi[i]))
        print()
        print("Transition Distribution")
        for i in range(N):
            print("  From Author {}".format(i))
            for j in range(N):
                print("    To Author {}: {:.4f}".format(j, A[i][j]))
        print()
        print("Emission Distribution")
        for i in range(N):
            print("  Actual Author {}".format(i))
            for j in range(N):
                print("    Observed Author {}: {:.4f}".format(j, B[i][j]))
        print()
        print()

