# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 11:10:07 2023

@author: hadaw
"""

from neuralNetwork import *

class InputLayer:
    def __init__(self, num_neurons):
        self.neurons = [Neuron(inputs=[random.uniform(-1, 1) for _ in range(num_neurons)],
                               weights=[random.uniform(-1, 1) for _ in range(num_neurons)],
                               bias=[random.uniform(-1, 1)])
                        for _ in range(num_neurons)]

        for neurons in self.neurons:
            self.weights = neurons.weights
            #print(self.weights)
            self.inputs = neurons.inputs

            self.bias = neurons.bias




    def forward_propagate(self, inputs):
            outputs = []
            for neuron in self.neurons:
                outputs.append(neuron.forward_propagate())
            return outputs


inputlayer = InputLayer(5)
sample_inputs = [0.5, 0.6, 0.7, 0.8, 0.9]
outputs = inputlayer.forward_propagate(sample_inputs)

print(inputlayer.neurons)
