""" created by colton hadaway 12/31/23 3:32AM"""
import math
import numpy as np
from decimal import Decimal
from mpmath import mp
mp.dps = 1000

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
    def __init__(self, layer_structure=[5,10,1], inputs=[], weights=[], bias=np.random.randint(0,9), e=False):
        if e == False:
            self.e = 1
            e=self.e
        else:
            self.e=e

        self.e = e
        self.layer_structure = layer_structure
        self.inputs = inputs
        self.bias = bias
        self.weights = weights
        self.c = 299792458


    def _e_(self):
        self.e = 1/math.factorial(self.e)
        return self.e

    def forward_propagate(self, activationFunction="sigmoid"):
        """
        Perform forward propagation.

        input_data: Input data to the network.
        """
        if activationFunction == "sigmoid":
            activationFunction = self.sigmoid
        else:
            pass

        activations = [self.inputs]

        for w, b in zip(self.weights, self.bias):
            z = np.dot(activations[-1], w) + b
            a = activationFunction(z)  # You can change this to any other activation function if needed.
            activations.append(a)
        return activations[-1]


    def prediction(self, x_i):
        # # Start with the input
        # output = x_i

        # # Pass through each hidden layer
        # for layer in self.hidden_layers:
        #     weighted_sum = np.dot(output, layer.weights) + layer.bias
        #     output = layer.activation_function(weighted_sum)

        # # Pass through the output layer (assuming no activation or linear activation)
        # weighted_sum = np.dot(output, self.output_layer.weights) + self.output_layer.bias
        # y_hat_i = weighted_sum  # Or use a linear activation function here if needed

        # return y_hat_i
        return 1

    def weighted_sum(self, net_type=("ff","rnn") ):
        if net_type == "ff":
            net_type = True
        else:
            net_type = False

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

        if net_type == False:
            return x+self.bias
        else:
            return x


    """
    #####################################################
        Activation functions are defined below
    #####################################################
    """
    def sigmoid(self, x):
        print(self.e)
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
    """
    ############################################################
        Loss functions are defined below
    ############################################################
    """
    def MeanSquaredError(self, NumberOfSamples):
        n = 1/NumberOfSamples
        residualSquared = 0.0
        for i in range(0, NumberOfSamples):
            y_i = self.inputs[i]
            yHat_i = self.prediction(y_i)

            residualSquared+=(y_i-yHat_i)**2

        return n*(residualSquared/NumberOfSamples)

    def MeanAbsoluteError(self, NumberOfSamples):
        n = 1/NumberOfSamples  # Assign totalSamples to n
        absoluteDifferences = 0.0

        for i in range(NumberOfSamples):
            y_i = self.inputs[i]  # actual values
            yHat_i = self.prediction(y_i)  # Predicted value for the i-th sample

            absoluteDifferences+=abs(y_i - yHat_i)

        return n*(absoluteDifferences/NumberOfSamples)

    def BinaryCrossEntropyLoss(self, NumberOfSamples):
        n = -(1/NumberOfSamples)
        total_loss = 0.0

        for i in range(NumberOfSamples):
            y_i = self.inputs[i]
            yHat_i = self.sigmoid(y_i)
            print(f"y_i: {y_i} | yHat_i: {yHat_i} n: {n}")

            loss = ((y_i*np.log(yHat_i))+((1-y_i)*np.log(1-yHat_i)))

            total_loss += loss

        return n*(total_loss/NumberOfSamples)


    def CategoricalCrossEntropyLoss(self):
        pass

    def SparseCategoricalCrossEntropyLoss(self):
        pass

    def HingeLoss(self):
        pass

    def HuberLoss(self):
        pass

    def PoissonLoss(self):
        pass

    def KullbackLeiblerDivergence(self):
        pass

    def ContrastiveLoss(self):
        pass


Neuron = Neuron(inputs=[np.random.randint(0,9) for i in range(0,10)], weights=[np.random.randint(0,9) for i in range(0,10)],bias=[np.random.randint(0,9) for i in range(0,10)])
NWSum = Neuron.weighted_sum(net_type="ff")
forward = Neuron.forward_propagate()
# sigmoid = Neuron.sigmoid(NWSum)
# relu = Neuron.reLU(NWSum)
# leakyrelu = Neuron.leakyReLU(NWSum)
# tanh = Neuron.tanH(NWSum)
# softmax = Neuron.softmax(NWSum)

# mse = Neuron.BinaryCrossEntropyLoss(8)
# mae = Neuron.MeanSquaredError(8)
# bcs = Neuron.MeanAbsoluteError(8)
# print(f"|\nsigmoid: {sigmoid} \n    type: {type(sigmoid)} \nreLU: {relu}  \n    type: {type(relu)} \nleakyreLU: {leakyrelu} \n    type: {type(leakyrelu)} \ntanH: {tanh} \n    type: {type(tanh)} \nsoftmax: {softmax} \n    type: {type(softmax)} |")
# print(f"""
#    \n bcs: {bcs}
#    \n mae: {mae}
#    \n mse: {mse}
# """)

#print(Neuron.sigmoid(Neuron.weighted_sum(net_type="rnn")))
#e = Neuron._e_()
