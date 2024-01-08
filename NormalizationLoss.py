# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 10:46:46 2024

@author: hadaw
"""

import numpy as np
from scipy.stats import entropy
from mpmath import mp
from itertools import cycle

mp.dps = 10

class Normal:
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.size = layer.size

    def _normalize_(self, layer, function):
        return np.vectorize(function)(layer)
    """
    ############################################################
        Loss functions are defined below
    ############################################################
    """

    # Created by colton hadaway on 01/07/2024 8:33 PM
    def ContrastiveLoss(self, binaryLabels, DataPoints, margin=1.0):
        def compute_distance(point1, point2):
            """
            Compute the Euclidean distance between two points in the feature space.

            Parameters:
                - point1 (numpy array): First data point.
                - point2 (numpy array): Second data point.

            Returns:
                - float: Euclidean distance between the two points.
            """
            return np.linalg.norm(point1 - point2)

        loss = 0.0

        """
        Compute Contrastive Loss for pairs of data points based on the given labels.

        Parameters:
            - labels (list): List of binary labels (1: similar, 0: dissimilar).
            - data_points (list of numpy arrays): List of data points in the feature space.
            - margin (float): Margin value for Contrastive Loss.

        Returns:
            - list: Computed Contrastive Losses for each pair of data points.
        """
        # Create a cycle iterator for the data points
        data_cycle = cycle(DataPoints)

        # Calculate Contrastive Loss for each pair
        loss = 0.0
        for label in binaryLabels:
            point1 = next(data_cycle)
            point2 = next(data_cycle)

            if label == 1:
                loss += 0.5*(compute_distance(point1, point2)**2)
            else:
                loss += 0.5*(max(0, (margin-compute_distance(point1, point2)))**2)

        # We use datapoints because its the distance across space-time,
        # otherwise it could be the distance across point to point.
        contrastive_loss = loss
        binary_performance = loss/len(binaryLabels)
        dataPoints_performance = loss/len(DataPoints)
        OverallPerformance = 0.5*(binary_performance+dataPoints_performance)
        averageOverallPerformance = 0.5*((binary_performance+dataPoints_performance)/2)


        return (("contrastive_loss", contrastive_loss),
                ("binary_performance",binary_performance),
                ("dataPoints_performance",dataPoints_performance),
                ("OverallPerformance",OverallPerformance),
                ("averageOverallPerformance",averageOverallPerformance)
                )

    def MeanSquaredError(self, NumberOfSamples):
        """
        Compute the average Mean Squared Error (MSE) over a set of samples.

        The Mean Squared Error (MSE) is a commonly used metric to measure
        the average squared difference between the predicted and actual values.
        It provides a measure of the model's accuracy in terms of prediction error.

        Parameters
        ----------
        NumberOfSamples : int
            The number of samples or data points to be evaluated.

        Returns
        -------
        mse : float
            The average Mean Squared Error computed over all samples.

        Notes
        -----
        The Mean Squared Error for each sample is calculated as:
            (y_i - yHat_i)^2
        where y_i is the actual value and yHat_i is the predicted value
        obtained from the model's prediction function.

        Example
        -------
        >>> MeanSquaredError(5)
        0.25
        """

        # Calculate the normalization factor
        n = 1 / NumberOfSamples

        # Initialize the total squared residual to zero
        residualSquared = 0.0

        # Loop through each sample index
        for i in range(0, NumberOfSamples):

            # Retrieve the actual value for the current sample
            y_i = self.inputs[i]

            # Predict the value for the current sample using the model's prediction function
            yHat_i = self.prediction(y_i)

            # Compute and accumlate the squared residual for the current sample
            residualSquared += (y_i - yHat_i) ** 2

        # Compute the average MSE by dividing the total squared residual by the number of samples
        return n * (residualSquared / NumberOfSamples)


    def MeanAbsoluteError(self, NumberOfSamples):
        """
        Compute the average Mean Absolute Error (MAE) over a set of samples.

        The Mean Absolute Error (MAE) is a commonly used metric to measure
        the average absolute difference between the predicted and actual values.
        It provides a measure of the model's accuracy in terms of absolute error.

        Parameters
        ----------
        NumberOfSamples : int
            The number of samples or data points to be evaluated.

        Returns
        -------
        mae : float
            The average Mean Absolute Error computed over all samples.

        Notes
        -----
        The Mean Absolute Error for each sample is calculated as:
            |y_i - yHat_i|
            where y_i is the actual value and yHat_i is the predicted value
            obtained from the model's prediction function.

        Example
        -------
        >>> MeanAbsoluteError(5)
        0.25
        """

        # Calculate the normalization factor
        n = 1 / NumberOfSamples

        # Initialize the total absolute differences to zero
        absoluteDifferences = 0.0

        # Loop through each sample index
        for i in range(NumberOfSamples):

            # Retrieve the actual value for the current sample
            y_i = self.inputs[i]

            # Predict the value for the current sample using the model's prediction function
            yHat_i = self.prediction(y_i)

            # Compute and accumulate the absolute difference for the current sample
            absoluteDifferences += abs(y_i - yHat_i)

        # Compute the average MAE by dividing the total absolute differences by the number of samples
        return n * (absoluteDifferences / NumberOfSamples)


    def BinaryCrossEntropyLoss(self, NumberOfSamples):
        """
        Compute the average Binary Cross-Entropy loss over a set of samples.

        The Binary Cross-Entropy loss is commonly used in binary classification
        tasks to measure the difference between the predicted probabilities and
        the actual binary labels. It penalizes incorrect predictions more severely.

        Parameters
        ----------
        NumberOfSamples : int
            The number of samples or data points to be evaluated.

        Returns
        -------
        loss : float
            The average Binary Cross-Entropy loss computed over all samples.

        Notes
        -----
        The Binary Cross-Entropy loss for each sample is calculated as:
            -(y_i * log(yHat_i) + (1 - y_i) * log(1 - yHat_i))
        where y_i is the actual binary label (0 or 1) and yHat_i is the predicted
        probability, typically obtained from a sigmoid function.

        Example
        -------
        >>> BinaryCrossEntropyLoss(5)
        0.25
        """

        # Calculate the normalization factor
        n = -(1 / NumberOfSamples)

        # Initialize the total loss to zero
        total_loss = 0.0

        # Loop through each sample index
        for i in range(NumberOfSamples):

            # Retrieve the actual binary label for the current sample (usually 0 or 1)
            y_i = self.inputs[i]

            # Predict the probability for the current sample using the sigmoid function
            yHat_i = self.sigmoid(y_i)

            # Compute and accumulate the Binary Cross-Entropy loss for the current sample
            total_loss += (y_i * mp.log(yHat_i)) + ((1 - y_i) * mp.log(1 - yHat_i))


        # Compute the average loss by dividing the total loss by the number of samples
        return n * (total_loss / NumberOfSamples)


    def CategoricalCrossEntropyLoss(self, true_labels, predicted_probabilities):
        """
        Compute the Categorical Cross-Entropy Loss.

        Parameters:
            - true_labels: The true labels in one-hot encoded format.
            - predicted_probabilities: The predicted probabilities for each class.

        Returns:
            - loss: The computed cross-entropy loss.
        """

        # Add a small constant to avoid logarithm of zero
        epsilon = 1e-10

        # Compute the cross-entropy loss
        loss = -np.sum(true_labels * np.log(predicted_probabilities + epsilon))

        # Normalize the loss by the number of samples
        loss /= len(true_labels)

        return loss


    def SparseCategoricalCrossEntropyLoss(self, true_labels, predicted_probabilities):
        """
        Compute the Sparse Categorical Cross-Entropy Loss.

        This function computes the loss between the true class labels and the predicted
        probabilities using the Sparse Categorical Cross-Entropy Loss formula. It accounts
        for potential division by zero by adding a small epsilon value to the logarithm.

        Parameters:
            -----------
            true_labels : numpy.ndarray
            The true class labels. Expected to be integers representing the correct class indices.

        predicted_probabilities : numpy.ndarray
            The raw (non-normalized) scores or logits for each class.

        Returns:
            --------
            float
            The computed Sparse Categorical Cross-Entropy Loss averaged over the batch.

        """

        # Add a small constant to avoid logarithm of zero
        epsilon = 1e-10

        # Calculate the negative log probabilities for the true labels
        # Convert the predicted probabilities to softmax values to ensure they are probabilities
        normalized_probabilities = self.softmax(predicted_probabilities)

        # Compute the loss using the formula: -sum(true_labels * log(predicted_probabilities + epsilon))
        loss = -np.sum(true_labels * np.log(normalized_probabilities + epsilon))

        # Average the loss over the number of samples
        loss /= len(true_labels)

        return loss


    def HingeLoss(self, NumberOfSamples):
        """
        Compute the average Hinge loss over a set of samples.

        The Hinge loss is commonly used in binary classification tasks
        with support vector machines (SVMs). It measures the maximum margin
        between the decision boundary and the samples, penalizing misclassified
        points more severely.

        Parameters
        ----------
        NumberOfSamples : int
            The number of samples or data points to be evaluated.

        Returns
        -------
        loss : float
            The average Hinge loss computed over all samples.

        Notes
        -----
        The Hinge loss for each sample is calculated as:
            max(0, 1 - (y_i * yHat_i))
            where y_i is the actual label (-1 or 1) and yHat_i is the predicted
            score, typically obtained from a sigmoid function.

        Example
        -------
        >>> HingeLoss(5)
        0.25
        """

        # Initialize the loss to zero
        loss = 0.0

        # Loop through each sample index
        for i in range(NumberOfSamples):

            # Retrieve the actual label for the current sample (usually -1 or 1)
            y_i = self.inputs[i]

            # Predict the score for the current sample using the sigmoid function
            yHat_i = self.sigmoid(y_i)

            # Compute and accumulate the Hinge loss for the current sample
            loss += np.max(0, 1 - (y_i * yHat_i))

        # Compute the average loss by dividing the total loss by the number of samples
        return loss / NumberOfSamples


    def HuberLoss(self, NumberOfSamples, threshold=float):
        """
        Compute the average Huber loss over a set of samples, given a threshold.

        The Huber loss combines properties of the mean squared error (MSE)
        and the mean absolute error (MAE). It behaves like the MAE for small
        errors and like the MSE for large errors, providing a balance between
        robustness and sensitivity to outliers.

        Parameters
        ----------
        NumberOfSamples : int
            The number of samples or data points to be evaluated.

        threshold : float, optional
            The threshold value that differentiates between the quadratic
            (MSE-like) and linear (MAE-like) parts of the loss function.
            Default is set to float(threshold).

        Returns
        -------
        loss : float
            The average Huber loss computed over all samples.

        Notes
        -----
        The Huber loss is defined as:
            - (1/2) * (difference^2)                  if |difference| <= threshold
            - threshold * (|difference| - (1/2) * threshold)  if |difference| > threshold
        Where difference is the absolute difference between the actual
        value y_i and the predicted value yHat_i for each sample.

        Example
        -------
        >>> HuberLoss(5, threshold=1.5)
        0.25
        """

        # Initialize the loss to zero
        loss = 0.0

        # Convert the threshold to float (in case it's passed as a different type)
        threshold = float(threshold)

        # Loop through each sample index
        for i in range(NumberOfSamples):

            # Retrieve the actual value for the current sample
            y_i = self.inputs[i]

            # Predict the value for the current sample using the provided prediction function
            yHat_i = self.prediction(y_i)

            # Calculate the absolute difference between the actual and predicted values
            difference = abs(y_i - yHat_i)

            # Determine the type of loss based on the difference and threshold
            if difference <= threshold:
                # MSE-like quadratic loss
                loss += ((1/2) * (difference**2))
            else:
                # MAE-like linear loss
                loss += (threshold * (difference - ((1/2) * threshold)))

        # Compute the average loss by dividing the total loss by the number of samples
        return loss / NumberOfSamples


    def PoissonLoss(self, EventInterval, TotalEvents):
        """
        Calculate the average Poisson loss based on the given event interval and total events.

        The Poisson loss quantifies the discrepancy between the observed number of events
        and the predicted number of events using the Poisson distribution. A lower loss
        indicates a better fit of the model to the data.

        Parameters
        ----------
        EventInterval : int or float
            The duration or interval over which events are observed.

        TotalEvents : list
            A list containing the actual counts of events observed during the specified interval.

        Returns
        -------
        loss : float
            The average Poisson loss computed over all observed events.
            A lower value indicates a better fit of the model to the data.

        Notes
        -----
        The Poisson distribution is a probability distribution that describes the number
        of events occurring in a fixed interval of time or space. The expected number
        of events (lambda, λ) in the Poisson distribution is the average rate at which
        events occur. The Poisson loss is calculated as the negative log-likelihood
        of the Poisson distribution for each observed event.

        Example
        -------
        >>> PoissonLoss(30, [25, 28, 32, 27])
        0.25
        """

        # Step 1: Calculate the average rate (lambda, λ) for the Poisson distribution
        Lambda = sum(TotalEvents) / len(TotalEvents)

        # Initialize the loss to zero
        loss = 0.0

        # Loop through each observed event count in TotalEvents
        for y in TotalEvents:
            # Step 2: Predicted value based on the Poisson distribution is the average rate
            yHat = Lambda

            # Step 3: Compute the Poisson loss for the current observed event count
            # Using the Poisson loss formula: yHat - (y * log(yHat)) + log(y!)
            # Where y! denotes the factorial of y
            loss += yHat - (y * np.log(yHat)) + np.log(mp.factorial(y))

        # Step 4: Compute the average loss by dividing the total loss by the number of events
        return loss / len(TotalEvents)

    """
    ############################################################
            Probability functions are defined below
    ############################################################
    """
    def averageClassicalProbability(self, value):
        return np.sum(np.linspace(0,abs(value)))/self.size


    def classicalProbability(self, value):
        num_outcomes = self.size
        return value/num_outcomes


    def Empirical_Experimental_Probability(self,FavorableOutcomes, NumberOfTrials):
        return FavorableOutcomes/NumberOfTrials


    def Subjective_Probability(self):
        """This one will be experimental,
        it will also take a while to put together
        considering i will have to make a language net
        """
        pass


    def Conditional_Probability(self):
        pass


    def Joint_Probability(self):
        pass


    def Marginal_Probability(self):
        pass


    def Bayesian_Probability(self):
        pass


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


    def quantitativeComparison(self, divergenceOne,divergenceTwo):
        average_divergence_1 = sum(divergenceOne)/len(divergenceOne)
        average_divergence_2 = sum(divergenceTwo)/len(divergenceTwo)

        if average_divergence_1 < average_divergence_2:
            return {"divergenceOneLesser": 0,
                    "divergenceTwoGreater": 1}
        else:
            return {"divergenceOneGreater": 1,
                    "divergenceTwoLesser": 0}


    def kullbackLeiblerDivergence(self, p, q, averageDivergence=False):
        # Ensure that both distributions have the same size
        if len(p) != len(q):
            raise ValueError("Distributions must have the same size for KL Divergence calculation.")

        entropy_distribution = entropy(p,q)


        if averageDivergence == True:
            return np.sum(entropy_distribution)/entropy_distribution.size
        else:
            return entropy_distribution





# List of labels indicating pairs of data points
labels = [1, 0, 1, 0]  # Example labels (1: similar, 0: dissimilar)

# # Corresponding data points in the feature space
# # Assume data_points is a list of numpy arrays representing the feature vectors
# data_points = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([2, 3, 4]), np.array([5, 6, 7])]

# Margin value for Contrastive Loss
margin_value = 1.0




matrix = np.array([[1,2,3],
                  [4,5,6],
                  [7,8,9]])

# matrixTwo = np.array([[12,22,33],
#                   [43,54,66],
#                   [7,88,9]])

normal = Normal(matrix)
normal.ContrastiveLoss(labels, matrix)
# normalTwo = Normal(matrixTwo)

# classicalProbDistribution = normal._normalize_(matrix, normal.classicalProbability)
# ExperimentDistribution = normalTwo._normalize_(matrixTwo, normal.classicalProbability)

# absClassicalProbability = normal._normalize_(classicalProbDistribution, normal.averageClassicalProbability)
# experimentProbDist = normalTwo._normalize_(ExperimentDistribution, normal.averageClassicalProbability)


# kullbackLeiblerDivergence = normal.kullbackLeiblerDivergence(classicalProbDistribution, absClassicalProbability)
# experimentDivergence = normal.kullbackLeiblerDivergence(ExperimentDistribution, experimentProbDist)

# comparison = normal.quantitativeComparison(kullbackLeiblerDivergence, experimentDivergence)
# i can then use the klieber thing to predict the next value with these two distributions


# probability = np.sum(absClassicalProbability)/normal.size
# # norm = 0.0

# # vector = np.vectorize(normal.ClassicalProbability)(matrix)
# # vec = []
# # for value in normal.iterMatrix():
# #     norm+= normal.ClassicalProbability(value)
# #     vec.append(norm)

# #print(abs(norm)/normal.size)
