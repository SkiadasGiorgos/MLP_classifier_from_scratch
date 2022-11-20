import random

import numpy as np

class Layer:
    def __init__(self, number_of_inputs, number_of_neurons, glorot):
        self.output = 0
        self.number_of_inputs = number_of_inputs
        self.number_of_neurons = number_of_neurons
        self.weights = np.zeros([number_of_inputs, self.number_of_neurons])
        if not glorot:
            self.weights = 0.1 * np.random.randn(number_of_inputs, self.number_of_neurons)
        else:
            self.glorot_weight_initialization()
        self.bias = np.zeros([1, number_of_neurons])
        self.weight_cache = np.zeros_like(self.weights)
        self.bias_cache = np.zeros_like(self.bias)

    def calculate_neuron_output(self, inputs):
        self.inputs = inputs
        self.output = np.dot(self.inputs, self.weights) + self.bias
        return self.output

    def backward(self, dvalues):
        self.dweights = np.dot(np.transpose(self.inputs), dvalues)
        self.dbias = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, np.transpose(self.weights))

    def glorot_weight_initialization(self):
        sd = np.sqrt(6.0/(self.number_of_inputs+self.number_of_neurons))
        for i in range(self.number_of_inputs):
            for j in range(self.number_of_neurons):
                x = np.float32(random.uniform(-sd,sd))
                self.weights[i][j] = x