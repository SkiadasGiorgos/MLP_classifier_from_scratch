import numpy as np
import random
from keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder

(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = y_train[0:100]
x_train = x_train[:100]
x_train = x_train.reshape(x_train.shape[0], -1)

def min_max_scaling(inputs):
    inputs_max = np.max(inputs)
    inputs_min = np.min(inputs)
    return np.divide(np.subtract(inputs,inputs_max), (inputs_max-inputs_min))

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
        self.weights = 0.1*np.random.randn(number_of_inputs, self.number_of_neurons)
        self.bias = np.zeros([1, number_of_neurons])

    def calculate_neuron_output(self, inputs):
        self.inputs = inputs
        self.output = np.dot(self.inputs, self.weights) + self.bias
        return self.output

    def backward(self,dvalues):
        self.dweights = np.dot(np.transpose(self.inputs),dvalues)
        self.dbias = np.sum(dvalues,axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues,np.transpose(self.weights))

class ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        return self.output

    def backward(self,dvalues):
        self.dvalues = dvalues.copy
        self.dvalues = np.sign(self.inputs)

class Softmax:

    def forward(self, inputs):
        # the subtraction helps with memory overflow (not mentioned in nn book)
        exponent = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        exponent_sum = np.sum(exponent, axis=1, keepdims=True)
        self.output = np.divide(exponent, exponent_sum)
        return self.output

    def backward(self,dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output,dvalues)):
            single_output = single_output.reshape(-1,1)
            jacobian = np.diagflat(single_output) - np.dot(single_output,single_output.T)
            self.dinputs[index] = np.dot(jacobian,single_dvalues)

class Loss:
    def categorical_cross_entropy(self, y_predict, y):
        samples = len(y)
        y_predict_clipped = np.clip(y_predict, 1e-7, 1 - 1e-7)
        correct_confidences = np.sum(y_predict_clipped * y, axis=1)
        log_likelihood = -np.log(correct_confidences)
        data_loss = np.mean(log_likelihood)
        return data_loss

    def categorical_cross_entropy_backward(self,dvalues,y):
        samples = len(dvalues)
        labels = len(dvalues[0])

        self.dinputs = -y/dvalues
        self.dinputs = self.dinputs/samples

class Softmax_loss_combination():
    def __init__(self):
        self.activation = Softmax()
        self.loss = Loss()

    def forward(self,inputs,y):
        self.activation.forward(inputs)
        self.ouput = self.activation.output

        return self.loss.categorical_cross_entropy(self.ouput,y)

    def backward(self,dvalues,y):
        sample = len(dvalues)

        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(sample), y] -= 1
        self.dinputs = self.dinputs/sample

class SGD_optimization():
    def __init__(self,learning_rate):
        self.learning_rate = learning_rate

    def update_parameters(self,layer):
        layer.weights += self.learning_rate*layer.dweights
        layer.bias += self.learning_rate*layer.dbias

def accuracy(y_pred,y):
    accuracy = np.mean(y_pred==y)
    return accuracy

soft = Softmax()
reLu = ReLU()
SGD = SGD_optimization(learning_rate=0.1)
soft_loss_combo = Softmax_loss_combination()
first_layer = Layer(784, 60)
second_layer = Layer(60, 60)
output_layer = Layer(60, 10)
X = x_train

for epochs in range(100):
    first_layer_output = first_layer.calculate_neuron_output(X)
    first_layer_activation = reLu.forward(first_layer_output)
    second_layer_output = second_layer.calculate_neuron_output(first_layer_activation)
    second_layer_activation = reLu.forward(second_layer_output)
    output_layer_output = output_layer.calculate_neuron_output(second_layer_activation)
    output_layer_activation = soft.forward(output_layer_output)
    loss = soft_loss_combo.forward(output_layer_activation, y_train)
    acc = accuracy(output_layer_activation, y_train)
    print("Epoch:",end=" ")
    print(epochs,end=" ")
    print("Loss:",end=" ")
    print(loss,end=" ")
    print("Accuracy:",end=" ")
    print(acc)
    ## Backward pass
    soft_loss_combo.backward(soft_loss_combo.ouput, y_train)
    output_layer.backward(soft_loss_combo.dinputs)
    second_layer.backward(output_layer.dinputs)
    first_layer.backward(second_layer.dinputs)

    ##Optimization

    SGD.update_parameters(output_layer)
    SGD.update_parameters(second_layer)
    SGD.update_parameters(first_layer)

    print(first_layer.weights)
    print(first_layer.bias)



# soft_output = np.array([[0.7,0.1,0.2],[0.1,0.5,0.4],[0.02,0.9,0.08]])
# targ = np.array([0,1,1])
#
# soft_loss = softmax_loss_combination()
# soft_loss.backward(soft_output,targ)
# dvalues1 = soft_loss.dinputs
# print(dvalues1)
