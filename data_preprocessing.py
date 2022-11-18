import numpy as np
import random

def min_max_scaling(inputs):
    inputs_max = np.max(inputs)
    inputs_min = np.min(inputs)
    return (np.divide(np.subtract(inputs, inputs_max), (inputs_max - inputs_min)) - 0.5) * 2


def one_hot(y, num_of_classes):
    ohs = np.zeros((y.shape[0], num_of_classes))
    for i, y in enumerate(y):
        ohs[i][y] = 1
    return ohs


def shuffle_inputs(x_input, y_input):
    number_of_indices = np.arange(0, x_input.shape[0], 1)
    random.shuffle(number_of_indices)
    x_output = x_input[number_of_indices]
    y_output = y_input[number_of_indices]
    return x_output, y_output
