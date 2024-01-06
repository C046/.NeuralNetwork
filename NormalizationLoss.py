import numpy as np
from scipy.stats import entropy

class Normal:
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.size = layer.size

    def _normalize_(self, layer, function):
        return np.vectorize(function)(layer)

    def absClassicalProbability(self, value):
        return np.sum(np.linspace(0,abs(value)))/self.size

    def classicalProbability(self, value):
        num_outcomes = self.layer.size
        return value/num_outcomes

    def iterMatrix(self):
        """
        Iterate through the values of a matrix.

        Parameters:
            - matrix: numpy array or iterable
            The matrix to iterate through.

        Yields:
            - int or float
            Each individual value in the matrix.
        """
        if isinstance(self.layer, np.ndarray):
        # If the input is a NumPy array, iterate through it directly
            for value in self.layer.flatten():
                yield value
        else:
            # If it's a generic iterable, iterate through rows and values
            for row in self.layer:
                for value in row:
                    yield value

    def kullbackLeiblerDivergence(self, p, q):
        return entropy(p, q)

from scipy.stats import entropy



matrix = np.array([[1,2,3],
                  [4,5,6],
                  [7,8,9]])

normal = Normal(matrix)
classicalProbDistribution = normal._normalize_(matrix, normal.classicalProbability)
absClassicalProbability = normal._normalize_(classicalProbDistribution, normal.absClassicalProbability)
kullbackLeiblerDivergence = normal.kullbackLeiblerDivergence(classicalProbDistribution, absClassicalProbability)
