# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 11:10:07 2023

@author: hadaw
"""

from Neurons import *
import random

class InputLayer:
    def __init__(self, num_neurons, inputs, weights, bias):
        self.neurons = [Neuron(inputs=inputs,
                               weights=weights,
                               bias=bias)
                        for _ in range(num_neurons)]
        #self.loss = [neuron.HingeLoss(neuron.NumSamples) for neuron in self.neurons]
        
    def forward_propagate(self, activation="sigmoid", floats=True):
        outputs = [neuron.forward_propagate(activationFunction=activation) for neuron in self.neurons]
        if floats == True:
            floats = []
            for i in range(len(outputs)):
                floats.append([float(II) for II in outputs[i]])

            return floats
        else:
            del floats

            return outputs






num_neurons = 5
sample_inputs = [0.5, 0.6, 0.7, 0.8, 0.9]
weights = [random.uniform(-1, 1) for _ in range(num_neurons)]
bias = [random.uniform(-1, 1) for _ in range(num_neurons)]

inputlayer = InputLayer(num_neurons=num_neurons, inputs=sample_inputs, weights=weights, bias=bias)
outputs = inputlayer.forward_propagate()
print("loss: ", inputlayer.loss)
