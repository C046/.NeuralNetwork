""" created by colton hadaway 12/31/23 3:32AM"""
import random
import numpy as np
from mpmath import mp
from itertools import cycle
mp.dps = 10

"""
What to do at work:
    1.) Figure out why e
        simply is always == 1
"""

class Neuron:
    """
    Bias:
        Imagine you're trying to predict
        the price of a house. Even if all
        other factors are zero (like size,
        location, etc.), there's still a
        base price. This base price is like
        a "bias" in our network—it gives
        our predictions a starting point.
    """
    def __init__(self, layer_structure=np.matrix([]),ground_truth_labels=[], bias=np.random.randint(0,9), e=False):
        if e == False:

            self.e = sum(layer_structure)
            e=self.e
        else:
            self.e = self._e_()

        self.e = self._e_()
        self.layer_structure = layer_structure
        self.inputs = []
        self.bias = bias
        self.weights = []
        self.c = 299792458
        self.NumSamples = len(self.inputs)
        self.hidden_layers= []
        self.output_layers = []
        self.activation = self._NeuronActivatorTest_(ground_truth_labels)

    def _e_(self):
        self.e = 1/mp.factorial(self.e)
        return self.e

    def forward_propagate(self, activationFunction="sigmoid"):
        if activationFunction == "sigmoid":
            activationFunction = self.sigmoid
        else:
            pass

        # Ensure weights and bias are in list format for iteration
        if not isinstance(self.weights, list):
            self.weights = [self.weights]
        if not isinstance(self.bias, list):
            self.bias = [self.bias]

        # Check dimensions
        #assert len(self.weights) == len(self.bias), "Mismatch in dimensions of weights and biases."

        current_activation = self.inputs
        #i = 0
        for w, b in zip(self.weights, cycle(self.bias)):
           # i+=1
            # Debug print to see the values
            z = np.dot(current_activation, w) + b
            current_activation = activationFunction(z)

        return current_activation


    def prediction(self, x_i):
        # Start with the input
        output = x_i

        # Pass through each hidden layer
        for hiddenLayers in range(len(self.layer_structure)-1):
            self.hidden_layers.append(hiddenLayers)
            weighted_sum = np.dot(output, hiddenlayerWeights) + hiddenlayerbias
            output = self.activation_function(weighted_sum)

        weighted_sum_yHat_i = np.dot(output, self.output_layer.weights) + self.output_layer.bias

        return weighted_sum_yHat_i


    def weighted_sum(self, net_type=("ff","rnn") ):
        if net_type == "ff":
            net_type = False
        else:
            net_type = True

        #x = math.factorial(np.sum(self.inputs))
        # Storing the last known valid value
        last_valid_value = self.inputs
        try:
            # Attempting the factorial operation
            x = mp.factorial(np.sum(last_valid_value))
        except OverflowError:
            # In case of overflow, use the last known valid value
            x = last_valid_value

        for i in range(0,len(self.inputs)):
            if net_type == True:

                x = self.bias[i]+(x+(self.weights[i]*self.sigmoid(self.inputs[i])))
            else:



                #x = Decimal(str(x))


                # x = x/299792458
                # print(x)
                    x = x+(self.weights[i])*self.sigmoid(self.inputs[i])
                    net_type= False
        if net_type == False:
            return x+random.choice(self.bias)
        else:
            return x


    """
    #####################################################
        Activation functions are defined below
    #####################################################
    """
    def sigmoid(self, x):
        _x = 1+(self.e**-x)
        def __activation__():
            return 1/_x

        return __activation__()

    def reLU(self, x):
        return max(0, x)

    def leakyReLU(self, x):
        if x > 0:
            return x
        else:
            return 0.01 * x

    def tanH(self, x):
        return (((self.e**x)-(self.e)**-x) / ((self.e**x)+(self.e)**-x))

    def softmax(self, x):
        """
        Parameters
        ----------
        x : TYPE
            weighted sum of inputs for each class

        Returns
        -------
        softmax_values : TYPE
            DESCRIPTION.

        """
        exp_values = [self.e**i for i in [x]]  # Compute the exponentials of each value in x
        sum_exp_values = sum(exp_values)     # Sum of all exponentials

        softmax_values = [i / sum_exp_values for i in exp_values]  # Normalize to get softmax values

        return softmax_values

    def _NeuronActivatorTest_(self, ground_truth_labels):
        "Experimental"
        def _predict_(activation, x):
            return activation(x)

        activations = [
            "softmax",
            "tanH",
            "leakyReLU",
            "reLU",
            "sigmoid"
        ]

        best_activation = None
        best_accuracy = -1

        for activation in activations:
            predictions = [_predict_(activation, x) for x in self.input_data]
            accuracy = self.compute_accuracy(predictions, ground_truth_labels)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_activation = activation

        return best_activation
    """
    ############################################################
        Probability functions are defined below
    ############################################################
    """
    def classical_probability(num_outcomes):
        if num_outcomes < 1:
            raise ValueError("Number of outcomes must be at least 1.")

        probability_per_outcome = 1 / num_outcomes
        probabilities = np.full(num_outcomes, probability_per_outcome)

        return probabilities

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

    def abs_classical_probability(self, num_outcomes):
        # I created this function
        # this is my math from my meditation
        # colton hadaway 1/5/2024
        probability = 1 / num_outcomes
        for i in range(2, num_outcomes):
            probability += i / num_outcomes

        return abs(probability) / num_outcomes

    def probabilityDistribution(self, layer):
        # Assuming each element in the matrix represents the number of outcomes
        # Calculate probabilities for each element using vectorized operations
        probabilities = np.vectorize(self.abs_classical_probability)(layer)

        return probabilities

    """
    ############################################################
        Loss functions are defined below
    ############################################################
    """
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

    def probability_distribution(self, x):
        # x is supposed to be a random variable
        pass

        #_probability_distribution_()

        for i in range(len(self.inputs)):
            I = self.inputs[i]

        pass

    def ContrastiveLoss(self):
        pass

if __name__ == "__main__":
    pass
#Neuron = Neuron(inputs=[np.random.randint(0,9) for i in range(0,10)], weights=[np.random.randint(0,9) for i in range(0,10)],bias=[np.random.randint(0,9) for i in range(0,10)])
#NWSum = Neuron.weighted_sum(net_type="ff")
# forward = Neuron.forward_propagate()
# sigmoid = Neuron.sigmoid(NWSum)
# relu = Neuron.reLU(NWSum)
# leakyrelu = Neuron.leakyReLU(NWSum)
# tanh = Neuron.tanH(NWSum)
# softmax = Neuron.softmax(NWSum)
#hinge = Neuron.HingeLoss(Neuron.NumSamples)
#print(f"hinge loss: {hinge}")

# mse = Neuron.BinaryCrossEntropyLoss(8)
# mae = Neuron.MeanSquaredError(8)
# bcs = Neuron.MeanAbsoluteError(8)
# print(f"|\nsigmoid: {sigmoid} \n    type: {type(sigmoid)} \nreLU: {relu}  \n    type: {type(relu)} \nleakyreLU: {leakyrelu} \n    type: {type(leakyrelu)} \ntanH: {tanh} \n    type: {type(tanh)} \nsoftmax: {softmax} \n    type: {type(softmax)} |")
# print(f"""
#     \n bcs: {bcs}
#     \n mae: {mae}
#     \n mse: {mse}
# """)

# print(Neuron.sigmoid(Neuron.weighted_sum(net_type="rnn")))
# e = Neuron._e_()
