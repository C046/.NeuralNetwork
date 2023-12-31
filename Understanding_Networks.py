""" created by colton hadaway 12/31/23 3:32AM"""
import math
import numpy as np
from decimal import Decimal
from mpmath import mp
mp.dps = 1000


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
        self.c = 299792458


    def _e_(self):
        self.e = 1/math.factorial(self.e)
        return self.e

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
            if net_type == False:

                x = self.bias+(x+(self.weights[i]*self.sigmoid(self.inputs[i])))
            else:

                #x = Decimal(str(x))


                # x = x/299792458
                # print(x)
                x = x+(self.weights[i])*self.sigmoid(self.inputs[i])

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
    """
    ############################################################
        Loss functions are defined below
    ############################################################
    """
    def MeanSquaredError(self, totalSamples):
        n = 1/totalSamples
        residualSquared = 0.0
        for i in range(0, totalSamples):
            y_i = self.inputs[i]
            yHat_i = self.prediction(y_i)

            residualSquared+=(y_i-yHat_i)**2

        return residualSquared/totalSamples

    def MeanAbsoluteError(self, totalSamples):
        n = 1/totalSamples  # Assign totalSamples to n
        absoluteDifferences = 0.0

        for i in range(totalSamples):
            y_i = self.inputs[i]  # actual values
            yHat_i = self.prediction(y_i)  # Predicted value for the i-th sample

            absoluteDifferences+=abs(y_i - yHat_i)

        return n*absoluteDifferences

    def BinaryCrossEntropyLoss(self, NumberOfSamples):
        n = -(1/NumberOfSamples)
        total_loss = 0.0

        for i in range(NumberOfSamples):
            y_i = self.inputs[i]
            yHat_i = self.prediciton(y_i)

            loss = ((y_i*np.log(yHat_i))+((1-y_i)*np.log(1-yHat_i)))

            total_loss += loss

        return n*(total_loss/n)


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
