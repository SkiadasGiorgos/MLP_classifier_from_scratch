import numpy as np
import random
from keras.datasets import mnist
from layers import Layer
from activation_functions import Softmax, ReLU
from loss_functions import Loss, Softmax_loss_combination
from optimization_functions import RMSProp_optimizer, SGD_optimization_mommentum, SGD_optimization
from accuracy_fucntion import Accuracy
from data_preprocessing import one_hot, shuffle_inputs, min_max_scaling

(x_train, y_train), (x_test, y_test) = mnist.load_data()


def neuron_forward_pass(input_data, layer, activation_functions):
    layer_output = layer.calculate_neuron_output(input_data)
    layer_activation = activation_functions.forward(layer_output)
    return layer_activation


x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

classes = 10
y_train = one_hot(y_train, 10)
y_test = one_hot(y_test, 10)
x_train = min_max_scaling(x_train)
x_test = min_max_scaling(x_test)

soft = Softmax()
optimizer = RMSProp_optimizer(learning_rate=1e-3, rho=0.99, decay=False, decay_rate=1e-4)
# optimizer = SGD_optimization(learning_rate=1)
accuracy = Accuracy()

soft_loss_combo = Softmax_loss_combination()

first_layer = Layer(784, 256, glorot=False)
activation1 = ReLU()
second_layer = Layer(256, 128, glorot=False)
activation2 = ReLU()
third_layer = Layer(128, 128, glorot=False)
activation3 = ReLU()
output_layer = Layer(128, 10, glorot=False)
batch_size = 128
epochs = 100

number_of_runs = int(x_train.shape[0] / batch_size)

for epoch in range(epochs):
    x_train, y_train = shuffle_inputs(x_train, y_train)
    accuracy.batch_accuracy = []

    for step in range(number_of_runs):
        x_train_batch = x_train[step * (batch_size + 1):(step * (batch_size + 1) + batch_size):1, :]
        y_train_batch = y_train[step * (batch_size + 1):(step * (batch_size + 1) + batch_size), :]
        # First layer inputs don't follow the same principal as the rest, since every other layer
        # takes as input the output of the previous one
        first_layer.calculate_neuron_output(inputs=x_train_batch)
        activation1.forward(first_layer.output)
        second_layer.calculate_neuron_output(inputs=activation1.output)
        activation2.forward(second_layer.output)
        third_layer.calculate_neuron_output(inputs=activation2.output)
        activation3.forward(third_layer.output)
        output_layer.calculate_neuron_output(activation3.output)

        loss = soft_loss_combo.forward(output_layer.output, y_train_batch)
        predictions = np.argmax(soft_loss_combo.output, axis=1)
        y = np.argmax(y_train_batch, axis=1)
        accuracy.calculate_accuracy(predictions, y)
        ## Backward pass

        soft_loss_combo.backward(soft_loss_combo.output, y)
        output_layer.backward(soft_loss_combo.dinputs)
        activation3.backward(output_layer.dinputs)
        third_layer.backward(activation3.dinputs)
        activation2.backward(third_layer.dinputs)
        second_layer.backward(activation2.dinputs)
        activation1.backward(second_layer.dinputs)
        first_layer.backward(activation1.dinputs)
        #
        # for i in range(len(layers)-1,-1,-1):
        #     # Output layer inputs don't follow the same principal as the rest, since every other layer
        #     # takes as input the output of the next one
        #     if i == len(layers)-1:
        #         input = soft_loss_combo.dinputs
        #         pass
        #     else:
        #         input = layers[i+1].dinputs
        #
        #     layers[i].backward(input)

        ##Optimization

        optimizer.update_parameters(layer=first_layer)
        optimizer.update_parameters(layer=second_layer)
        optimizer.update_parameters(layer=third_layer)
        optimizer.update_parameters(layer=output_layer)

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
    epoch_loss = soft_loss_combo.epoch_loss()
    epoch_accuracy = accuracy.epoch_accuracy_update()
    print("Epoch:", end=" ")
    print(epoch, end=" ")
    print("Loss:", end=" ")
    print(loss, end=" ")
    print("Accuracy:", end=" ")
    print(epoch_accuracy)
