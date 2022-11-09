import numpy as np
import random
from keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder

(x_train, y_train), (x_test, y_test) = mnist.load_data()


y_train = y_train[0:100]
x_train = x_train[:100]
x_train = x_train.reshape(x_train.shape[0],-1)

def one_hot(y,num_of_classes):
    ohs = np.zeros((y.shape[0],num_of_classes))
    for i, y in enumerate(y):
        ohs[i][y] = 1
    return ohs

classes = 10
y_train = one_hot(y_train,10)

class Layer:
    def __init__(self, inputs,number_of_inputs, number_of_neurons):
        self.inputs = inputs
        self.output = 0
        self.number_of_inputs = number_of_inputs
        self.number_of_neurons = number_of_neurons
        self.weights = np.random.rand(number_of_inputs, self.number_of_neurons)
        self.bias = np.zeros([1, number_of_neurons])

    def calculate_neuron_output(self):
        self.output = np.dot(self.inputs, self.weights) + self.bias
        return self.output


class ReLU:
    def relu(self, inputs, derivative):
        if derivative:
            self.backprop = np.sign(inputs)
            output = self.backprop
        else:
            self.forward = np.maximum(0, inputs)
            output = self.forward
        return output


class Softmax:
    def softmax(self, inputs,derivative):
        # the subtraction helps with memory overflow (not mentioned in nn book)
        exponent = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        exponent_sum = np.sum(exponent, axis=1, keepdims=True)
        if derivative:
            self.output = np.divide(exponent,exponent_sum)*(1-np.divide(exponent,exponent_sum))
        else:
            self.output = np.divide(exponent, exponent_sum)
        return self.output

class Loss:
    def categorical_cross_entropy(self,y_predict,y):
        samples = len(y)
        y_predict_clipped = np.clip(y_predict,1e-7,1-1e-7)
        correct_confidences = np.sum(y_predict_clipped*y,axis=1)
        log_likelihood = -np.log(correct_confidences)
        data_loss = np.mean(log_likelihood)
        return data_loss

soft = Softmax()

reLu = ReLU()

first_layer = Layer(x_train, 784, 60)
first_layer_output = first_layer.calculate_neuron_output()
first_layer_activation = reLu.relu(first_layer_output,True)
second_layer = Layer(first_layer_activation, 60,60)
second_layer_output = second_layer.calculate_neuron_output()
second_layer_activation = reLu.relu(second_layer_output,derivative = False)
output_layer = Layer(second_layer_activation, 60,10)
output_layer_output = output_layer.calculate_neuron_output()
output_layer_activation = soft.softmax(output_layer_output,derivative = False)
print(output_layer_activation)

loss_function = Loss()
loss = loss_function.categorical_cross_entropy(output_layer_activation,y_train)
print(loss)


