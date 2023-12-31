import math
import numpy as np

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
    def __init__(self, inputs=[], weights=[], bias=np.random.randint(0,9), e=False):
        if e == False:
            self.e = 1
            e=self.e
        else:
            self.e=e

        self.e = e
        self.inputs = inputs
        self.bias = bias
        self.weights = weights


    def _e_(self):
        self.e = 1/math.factorial(self.e)
        return self.e

    def weighted_sum(self, net_type=("ff","rnn") ):
        if net_type == "ff":
            net_type = True
        else:
            net_type = False

        x = math.factorial(np.sum(self.inputs))

        for i in range(0,len(self.inputs)):
            if net_type == False:
                x = self.bias+(x+(self.weights[i]*self.sigmoid(self.inputs[i])))
            else:
                x = x+(self.weights[i]*self.sigmoid(self.inputs[i]))

        if net_type == False:
            return x
        else:
            return x+self.bias


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
        
