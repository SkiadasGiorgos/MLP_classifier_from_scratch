import numpy as np
from activation_functions import Softmax


class Loss:
    def categorical_cross_entropy(self, y_predict, y):
        samples = len(y)
        y_predict_clipped = np.clip(y_predict, 1e-7, 1 - 1e-7)
        correct_confidences = np.sum(y_predict_clipped * y, axis=1)
        log_likelihood = -np.log(correct_confidences)
        data_loss = np.mean(log_likelihood)
        return data_loss

    def categorical_cross_entropy_backward(self, dvalues, y):
        samples = len(dvalues)
        labels = len(dvalues[0])

        self.dinputs = -y / dvalues
        self.dinputs = self.dinputs / samples


class Softmax_loss_combination():
    def __init__(self):
        self.activation = Softmax()
        self.loss = Loss()

    def forward(self, inputs, y):
        self.activation.forward(inputs)
        self.output = self.activation.output

        return self.loss.categorical_cross_entropy(self.output, y)

    def backward(self, dvalues, y):
        sample = len(dvalues)

        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(sample), y] -= 1
        self.dinputs = self.dinputs / sample

