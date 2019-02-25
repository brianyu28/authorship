"""
Regression model to handle adjacency.
"""

from sklearn.linear_model import LogisticRegression

class AdjacencyModel:

    def __init__(self, non_adjacent, adjacent):
        x = [value for value in non_adjacent + adjacent]
        y = [False for _ in non_adjacent] + [True for _ in adjacent]
        reg = LogisticRegression()
        reg.fit(x, y)
        self.reg = reg

    def proba(self, observed):
        return self.reg.predict_proba([observed])[0][1]
