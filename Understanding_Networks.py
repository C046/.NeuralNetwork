import math
import numpy as np
""" created by colton hadaway 12/31/23 3:32AM"""
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
    def __init__(self, inputs=[], weights=[], bias=1, e=False):
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
            
    def sigmoid(self, x):
        _x = (self.e**-x)+1
        def __activation__():
            return 1/_x
            
        return __activation__()


Neuron = Neuron(inputs=[1,2,3,4], weights=[4,3,2,1])
print(Neuron.sigmoid(Neuron.weighted_sum(net_type="ff")))
#e = Neuron._e_()
