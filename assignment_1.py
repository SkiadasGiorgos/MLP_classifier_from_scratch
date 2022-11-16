import numpy as np
import random
from keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], -1)


def min_max_scaling(inputs):
    inputs_max = np.max(inputs)
    inputs_min = np.min(inputs)
    return (np.divide(np.subtract(inputs, inputs_max), (inputs_max - inputs_min)) - 0.5) * 2


def one_hot(y, num_of_classes):
    ohs = np.zeros((y.shape[0], num_of_classes))
    for i, y in enumerate(y):
        ohs[i][y] = 1
    return ohs


classes = 10
y_train = one_hot(y_train, 10)
x_train = min_max_scaling(x_train)


class Layer:
    def __init__(self, number_of_inputs, number_of_neurons):
        self.output = 0
        self.number_of_inputs = number_of_inputs
        self.number_of_neurons = number_of_neurons
        self.weights = 0.1 * np.random.randn(number_of_inputs, self.number_of_neurons)
        self.bias = np.zeros([1, number_of_neurons])

    def calculate_neuron_output(self, inputs):
        self.inputs = inputs
        self.output = np.dot(self.inputs, self.weights) + self.bias
        return self.output

    def backward(self, dvalues):
        self.dweights = np.dot(np.transpose(self.inputs), dvalues)
        self.dbias = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, np.transpose(self.weights))


class ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        return self.output

    def backward(self, dvalues):
        self.dvalues = dvalues.copy
        self.dvalues = np.sign(self.inputs)


class Softmax:

    def forward(self, inputs):
        # the subtraction helps with memory overflow (not mentioned in nn book)
        exponent = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        exponent_sum = np.sum(exponent, axis=1, keepdims=True)
        self.output = np.divide(exponent, exponent_sum)
        return self.output

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian, single_dvalues)


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
        self.ouput = self.activation.output

        return self.loss.categorical_cross_entropy(self.ouput, y)

    def backward(self, dvalues, y):
        sample = len(dvalues)

        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(sample), y] -= 1
        self.dinputs = self.dinputs / sample


class SGD_optimization():
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update_parameters(self, layer):
        layer.weights += self.learning_rate * layer.dweights
        layer.bias += self.learning_rate * layer.dbias


def accuracy(y_pred, y):
    accuracy = np.mean(y_pred == y)
    return accuracy


def shuffle_inputs(x_input, y_input):
    number_of_indices = np.arange(0, x_input.shape[0], 1)
    random.shuffle(number_of_indices)
    x_output = x_input[number_of_indices]
    y_output = y_input[number_of_indices]
    return x_output, y_output


soft = Softmax()
reLu = ReLU()
SGD = SGD_optimization(learning_rate=0.001)
soft_loss_combo = Softmax_loss_combination()
first_layer = Layer(784, 60)
second_layer = Layer(60, 128)
third_layer = Layer(128, 256)
output_layer = Layer(256, 10)
batch_size = 1000

number_of_runs = int(x_train.shape[0] / batch_size)

for epochs in range(100):
    x_train, y_train = shuffle_inputs(x_train, y_train)
    for step in range(number_of_runs):
        x_train_batch = x_train[step * (batch_size + 1):(step * (batch_size + 1) + batch_size):1, :]
        y_train_batch = y_train[step * (batch_size + 1):(step * (batch_size + 1) + batch_size), :]
        first_layer_output = first_layer.calculate_neuron_output(x_train_batch)
        first_layer_activation = reLu.forward(first_layer_output)
        second_layer_output = second_layer.calculate_neuron_output(first_layer_activation)
        second_layer_activation = reLu.forward(second_layer_output)
        third_layer_output = third_layer.calculate_neuron_output(second_layer_activation)
        third_layer_activation = reLu.forward(third_layer_output)
        output_layer_output = output_layer.calculate_neuron_output(third_layer_activation)
        output_layer_activation = soft.forward(output_layer_output)
        loss = soft_loss_combo.forward(output_layer_activation, y_train_batch)
        acc = accuracy(output_layer_activation, y_train_batch)

        ## Backward pass
        soft_loss_combo.backward(soft_loss_combo.ouput, y_train_batch)
        output_layer.backward(soft_loss_combo.dinputs)
        third_layer.backward(output_layer.dinputs)
        second_layer.backward(third_layer.dinputs)
        first_layer.backward(second_layer.dinputs)

        ##Optimization

        SGD.update_parameters(output_layer)
        SGD.update_parameters(second_layer)
        SGD.update_parameters(first_layer)
        SGD.update_parameters(third_layer)

    print("Epoch:", end=" ")
    print(epochs, end=" ")
    print("Loss:", end=" ")
    print(loss, end=" ")
    print("Accuracy:", end=" ")
    print(acc)
