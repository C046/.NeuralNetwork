""" created by colton hadaway 12/31/23 3:32AM"""
import random
import numpy as np
from mpmath import mp
from itertools import cycle
from NormalizationLoss import *
mp.dps = 10

class Neuron(Normal):
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
    def __init__(self, layer_structure=np.matrix([[1,5,1],
                                                 [2,5,1]]),ground_truth_labels=[], weights=np.array([]), bias=np.random.randint(0,9), e=False):
        # if e is false, perform a matrix sum
        if e == False:
            # This is needed, I figured it out a while ago but
            # To be real with you I dont remember.
            self.e = np.sum(layer_structure)
            e=self.e
        else:
            self.e = self._e_()


        # Make e accomodate to the model not via the model accomodate to e
        self.e = self._e_()
        # Initialize the input, hidden, and output layers based on the layer_structure matrix
        self.inputs = np.array([layer_structure[i, 0] for i in range(layer_structure.shape[0])])
        print("Inputs:", self.inputs)

        self.hiddenLayers = np.array([layer_structure[i, 1] for i in range(layer_structure.shape[0])])
        print("Hidden Layers:", self.hiddenLayers)

        self.output = np.array([layer_structure[i, 2] for i in range(layer_structure.shape[0])])
        print("Output:", self.output)


        # Init the weights
        self.weights = weights
        # Init the bias
        self.bias = bias

        # Init the speed of light, I do not know why yet but it feels right.
        self.c = 299792458

        self.NumSamples = self.inputs.size

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
            self.softmax,
            self.tanH,
            self.leakyReLU,
            self.reLU,
            self.sigmoid
        ]

        best_activation = None
        best_accuracy = -1

        for activation in activations:
            predictions = [_predict_(activation, x) for x in self.inputs]
            accuracy = self.compute_accuracy(predictions, ground_truth_labels)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_activation = activation

        return best_activation

    def get_layer_structure_shape(self, layer_structure):
        """
        Obtain the shape of the layer_structure matrix or array.

        Parameters:
            - layer_structure: NumPy matrix or array

        Returns:
            - Shape of the layer_structure
        """
        return np.shape(layer_structure)

    def generate_matrix_from_shape(self, shape, random=False, random_float=False):
        """
        Generate a matrix with the specified shape.

        Parameters:
            - shape: Tuple specifying the shape of the matrix
            - random: Boolean flag indicating whether to initialize the matrix with random values (default: False)
            - random_float: Boolean flag indicating whether to generate random values as floats (default: False)

        Returns:
            - Matrix with the specified shape initialized with zeros, random integers, or random floats
        """
        if random:
            if random_float:
                return np.random.rand(*shape)  # Initialize with random floats between 0 and 1
            else:
                return np.random.randint(0, 10, size=shape)  # Initialize with random integers between 0 and 9
        else:
            return np.zeros(shape)  # Initialize with zeros

if __name__ == "__main__":
    layer_structure=np.matrix([5,10,1])
    Neuron = Neuron()
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
