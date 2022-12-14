import numpy as np

class Accuracy():
    def __init__(self):
        self.accuracy = 0
        self.batch_accuracy = []
        self.epoch_accuracy = []
    def epoch_accuracy_update(self):
        epoch_acc = np.mean(self.batch_accuracy)
        self.epoch_accuracy.append(epoch_acc)
        return epoch_acc

    def calculate_accuracy(self, y_pred, y):
        if len(y.shape) == 2:
            y = np.argmax(y,axis = 1)
        self.accuracy = np.mean(y_pred == y)
        self.batch_accuracy.append(self.accuracy)

    def validation_accuracy(self, y_pred, y):
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        validation_accuracy = np.mean(y_pred == y)
        return validation_accuracy

    def new_pass(self):
        self.batch_accuracy = []

