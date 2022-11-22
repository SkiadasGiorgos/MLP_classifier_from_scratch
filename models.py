import numpy as np
from loss_functions import Softmax_loss_combination
class Model:
    def __init__(self):
        self.layers =[]

    def add(self,layer):
        self.layers.append(layer)

    def set(self,*,loss,optimizer,accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    def forward(self,x):
        self.input_layer.forward(x)

        for layer in self.layers:
            layer.forward(layer.prev.output)

        return layer.output

    def finalize_model(self):

        self.input_layer = Layer_Input()
        layer_count = len(self.layers)

        self.trainable_layers = []

        for i in range(layer_count):
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i + 1]
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]
            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

            self.loss.remember_trainable_layers(self.trainable_layers)

            self.softmax_classifier_output = Softmax_loss_combination()

    def train(self,x,y,*,epochs=1,print_every=1):
        for epoch in range(1,epochs+1):
            output = self.forward(x)

            data_loss = self.loss.calculate(output,y)
            reguralization_loss = self.loss.l2_regularization()
            loss = data_loss + reguralization_loss

            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate_accuracy(predictions,y)

            self.backward(output,y)

            for layer in self.trainable_layers:
                self.optimizer.update_parameters(layer)

            if not epoch%print_every:
                print(f'epoch:{epoch},'+
                      f'acc:{accuracy:},',
                      f'loss:{loss}')

    def backward(self,output,y):
        self.softmax_classifier_output.backward(output,y)
        self.layers[-1].dinputs = self.softmax_classifier_output.dinputs

        for layer in reversed(self.layers[:-1]):
            layer.backward(layer.next.dinputs)
        # self.loss.backward(output,y)


class Layer_Input():
    def forward(self,inputs):
        self.output = inputs
