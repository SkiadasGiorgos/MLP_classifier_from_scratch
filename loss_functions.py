import numpy as np
from activation_functions import Softmax

class Loss:
    def __init__(self):
        self.batch_loss = []
        self.epoch_loss = []
        self.batch_reguralization_loss = []
    def remember_trainable_layers(self,trainable_layers):
        self.trainable_layers = trainable_layers

    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        self.batch_loss.append(data_loss)
        return data_loss

    def validation_loss(self,output,y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss + self.l2_regularization()
    def l2_regularization(self):
        regularization_loss = 0
        for layer in self.trainable_layers:
            if layer.weight_regularizer > 0:
                regularization_loss += layer.weight_regularizer * np.sum(layer.weights * layer.weights)
            if layer.bias_regularizer > 0:
                regularization_loss += layer.bias_regularizer * np.sum(layer.bias * layer.bias)
            self.batch_reguralization_loss.append(regularization_loss)
        return regularization_loss

    def epoch_loss_update(self):
        if not self.batch_reguralization_loss:
           self.epoch_loss.append(np.mean(self.batch_loss))
        else:
            self.epoch_loss.append(np.mean(self.batch_loss)+np.mean(self.batch_reguralization_loss))

        return self.epoch_loss[-1]

    def new_pass(self):
        self.batch_loss = []

class CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples


class Softmax_loss_combination():
    def __init__(self):
        self.activation = Softmax()
        self.loss = CategoricalCrossEntropy()
    def backward(self, dvalues, y):
        sample = len(dvalues)

        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(sample), y] -= 1
        self.dinputs = self.dinputs / sample

