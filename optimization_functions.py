import numpy as np

class SGD_optimization():
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update_parameters(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.bias += -self.learning_rate * layer.dbias

class SGD_optimization_mommentum():
    def __init__(self, learning_rate, beta):
        self.learning_rate = learning_rate
        self.beta = beta
        self.updates = 0

    def update_parameters(self, layer):

        if self.updates == 0:
            self.weight_momentum = np.zeros_like(layer.weights)
            self.bias_momentum = np.zeros_like(layer.bias)
            self.updates += 1
        else:
            self.weight_momentum = layer.weights
            self.bias_momentum = layer.bias

        layer.weights += self.beta * self.weight_momentum - self.learning_rate * layer.dweights
        layer.bias += self.beta * self.bias_momentum - self.learning_rate * layer.dbias


class RMSProp_optimizer():

    def __init__(self, learning_rate, rho, decay, decay_rate):
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.rho = rho
        self.updates = 0
        self.epsilon = 1e-7  # helps with gradient explosion
        self.decay = decay

        if self.decay:
            self.decay_rate = decay_rate
            self.iterations = 0

    def update_learning_rate(self):
        self.learning_rate = self.initial_learning_rate*(1/(1+self.decay_rate*self.iterations))

    def update_parameters(self, layer):
        if self.decay:
            self.update_learning_rate()
        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights ** 2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbias ** 2

        layer.weights += -self.learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.bias += -self.learning_rate * layer.dbias / (np.sqrt(layer.bias_cache) + self.epsilon)
        if self.decay:
            self.iterations += 1