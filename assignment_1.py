import numpy as np
import random
from keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder

(x_train, y_train), (x_test, y_test) = mnist.load_data()



def min_max_scaling(inputs):
    inputs_max = np.max(inputs)
    inputs_min = np.min(inputs)
    return (np.divide(np.subtract(inputs, inputs_max), (inputs_max - inputs_min)) - 0.5) * 2


def one_hot(y, num_of_classes):
    ohs = np.zeros((y.shape[0], num_of_classes))
    for i, y in enumerate(y):
        ohs[i][y] = 1
    return ohs

class Layer:
    def __init__(self, number_of_inputs, number_of_neurons):
        self.output = 0
        self.number_of_inputs = number_of_inputs
        self.number_of_neurons = number_of_neurons
        self.weights = 0.1 * np.random.randn(number_of_inputs, self.number_of_neurons)
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

        layer.weights = self.beta * self.weight_momentum - self.learning_rate * layer.dweights
        layer.bias = self.beta * self.bias_momentum - self.learning_rate * layer.dbias


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
    def __init__(self, learning_rate, rho):
        self.learning_rate = learning_rate
        self.rho = rho
        self.updates = 0
        self.epsilon = 1e-7  # helps with gradient explosion

    def update_parameters(self, layer):
        layer.weight_cache = self.rho + layer.weight_cache + (1 - self.rho) * layer.dweights ** 2
        layer.bias_cache = self.rho + layer.bias_cache + (1 - self.rho) * layer.dbias ** 2

        layer.weights += self.learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.bias += self.learning_rate * layer.dbias / (np.sqrt(layer.bias_cache) + self.epsilon)


def accuracy(y_pred, y):
    accuracy = np.mean(y_pred == y)
    return accuracy


def shuffle_inputs(x_input, y_input):
    number_of_indices = np.arange(0, x_input.shape[0], 1)
    random.shuffle(number_of_indices)
    x_output = x_input[number_of_indices]
    y_output = y_input[number_of_indices]
    return x_output, y_output
def neuron_forward_pass(input_data,layer,activation_functions):
    layer_output = layer.calculate_neuron_output(input_data)
    layer_activation = activation_functions.forward(layer_output)
    return layer_activation

x_train = x_train.reshape(x_train.shape[0], -1)
x_test =  x_test.reshape(x_test.shape[0], -1)
classes = 10
y_train = one_hot(y_train, 10)
y_test = one_hot(y_test,10)
x_train = min_max_scaling(x_train)
x_test = min_max_scaling(x_test)

soft = Softmax()
reLu = ReLU()
SGD = RMSProp_optimizer(learning_rate=.001, rho=0.001)
soft_loss_combo = Softmax_loss_combination()
first_layer = Layer(784, 60)
second_layer = Layer(60, 128)
third_layer = Layer(128, 256)
fourth_layer = Layer(256,256)
output_layer = Layer(256, 10)
batch_size = 1000
epochs = 100


layers = [first_layer,second_layer,third_layer,output_layer]
activation_functions = [reLu,reLu,reLu,soft]
#
# layers = []
# activation_functions = []
#
#
# layers.append(Layer(784, 60))
# layers.append(Layer(60, 128))
# layers.append(Layer(128, 256))
# layers.append(Layer(256, 10))
# activation_functions.append(reLu)
# activation_functions.append(reLu)
# activation_functions.append(reLu)
# activation_functions.append(Softmax)

number_of_runs = int(x_train.shape[0] / batch_size)

for epoch in range(epochs):
    x_train, y_train = shuffle_inputs(x_train, y_train)
    for step in range(number_of_runs):
        x_train_batch = x_train[step * (batch_size + 1):(step * (batch_size + 1) + batch_size):1, :]
        y_train_batch = y_train[step * (batch_size + 1):(step * (batch_size + 1) + batch_size), :]
        # First layer inputs don't follow the same principal as the rest, since every other layer
        # takes as input the output of the previous one
        layers[0].inputs = x_train_batch

        for i in range(len(layers)):
            if i == 0:
                layers[i].inputs = x_train_batch
            else:
                layers[i].inputs = activation_functions[i-1].output
            neuron_forward_pass(layers[i].inputs,layers[i],activation_functions[i])

        loss = soft_loss_combo.forward(activation_functions[-1].output, y_train_batch)
        acc = accuracy(activation_functions[-1].output, y_train_batch)

        ## Backward pass

        soft_loss_combo.backward(soft_loss_combo.ouput, y_train_batch)

        for i in range(len(layers)-1,-1,-1):
            # Output layer inputs don't follow the same principal as the rest, since every other layer
            # takes as input the output of the next one
            if i == len(layers)-1:
                input = soft_loss_combo.dinputs
                pass
            else:
                input = layers[i+1].dinputs

            layers[i].backward(input)

        ##Optimization

        for i in range(len(layers)):
            SGD.update_parameters(layers[i])
    #
    # first_layer_output = first_layer.calculate_neuron_output(x_test)
    # first_layer_activation = reLu.forward(first_layer_output)
    # second_layer_output = second_layer.calculate_neuron_output(first_layer_activation)
    # second_layer_activation = reLu.forward(second_layer_output)
    # third_layer_output = third_layer.calculate_neuron_output(second_layer_activation)
    # third_layer_activation = reLu.forward(third_layer_output)
    # output_layer_output = output_layer.calculate_neuron_output(third_layer_activation)
    # output_layer_activation = soft.forward(output_layer_output)
    # val_loss = soft_loss_combo.forward(output_layer_activation, y_test)
    # val_acc = accuracy(output_layer_activation, y_test)

    print("Epoch:", end=" ")
    print(epoch, end=" ")
    print("Loss:", end=" ")
    print(loss, end=" ")
    print("Accuracy:", end=" ")
    print(acc)