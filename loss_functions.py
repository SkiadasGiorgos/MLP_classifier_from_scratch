import numpy as np
from activation_functions import Softmax


# class Loss:
#     def __init__(self):
#         self.batch_loss = []
#         self.epoch_loss = []
#
#     def categorical_cross_entropy(self, y_predict, y):
#         samples = len(y)
#         y_predict_clipped = np.clip(y_predict, 1e-7, 1 - 1e-7)
#         correct_confidences = np.sum(y_predict_clipped * y, axis=1)
#         log_likelihood = -np.log(correct_confidences)
#         data_loss = np.mean(log_likelihood)
#         self.batch_loss.append(data_loss)
#         return data_loss
#
#     def categorical_cross_entropy_backward(self, dvalues, y):
#         samples = len(dvalues)
#         labels = len(dvalues[0])
#
#         self.dinputs = -y / dvalues
#         self.dinputs = self.dinputs / samples

class Loss:
    def __init__(self):
        self.batch_loss = []
        self.epoch_loss = []
    def calculate(self,output,y):
        sample_losses = self.forward(output,y)
        data_loss = np.mean(sample_losses)
        self.batch_loss.append(data_loss)
        return data_loss

class CategoricalCrossEntropy(Loss):
    def forward(self,y_pred,y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred,1e-7,1-1e-7)
        if len(y_true.shape)  == 1:
            correct_confidences = y_pred_clipped[range(samples),y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true,axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self,dvalues,y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true /dvalues
        self.dinputs = self.dinputs/samples

class Softmax_loss_combination():
    def __init__(self):
        self.activation = Softmax()
        self.loss = CategoricalCrossEntropy()

    def forward(self, inputs, y):
        self.activation.forward(inputs)
        self.output = self.activation.output

        return self.loss.calculate(self.output, y)

    def backward(self, dvalues, y):
        sample = len(dvalues)

        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(sample), y] -= 1
        self.dinputs = self.dinputs / sample

    def epoch_loss(self):
        self.loss.epoch_loss.append(np.mean(self.loss.batch_loss))
        return self.loss.epoch_loss[-1]


