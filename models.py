from loss_functions import Softmax_loss_combination
from data_preprocessing import shuffle_inputs
from matplotlib import pyplot as plt

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


    def validation(self, x_validation, y_validation, test=False):
        self.accuracy.new_pass()
        self.loss.new_pass()
        output = self.forward(x_validation)
        predictions = self.output_layer_activation.predictions(output)
        accuracy = self.accuracy.validation_accuracy(predictions,y_validation)
        loss = self.loss.validation_loss(output,y_validation)

        if test:
            print(
                  f'Test accuracy :{accuracy:},',
                  f'Test loss:{loss},'
                  )

        return loss, accuracy
    def train(self,x,y,*,epochs=1,print_every=1,batch_size,x_validation,y_validation):
        number_of_runs = x.shape[0]//batch_size
        x,y = shuffle_inputs(x,y)

        for epoch in range(1,epochs+1):
            self.loss.new_pass()
            self.accuracy.new_pass()
            for step in range(number_of_runs):
                x_batch = x[(step*batch_size):((step+1)*batch_size ),:]
                y_batch = y[(step*batch_size):((step+1)*batch_size ),:]
                output = self.forward(x_batch)

                data_loss = self.loss.calculate(output,y_batch)
                reguralization_loss = self.loss.l2_regularization()
                loss = data_loss + reguralization_loss

                predictions = self.output_layer_activation.predictions(output)
                self.accuracy.calculate_accuracy(predictions,y_batch)

                self.backward(output,y_batch)

                for layer in self.trainable_layers:
                    self.optimizer.update_parameters(layer)


            accuracy = self.accuracy.epoch_accuracy_update()
            loss = self.loss.epoch_loss_update()

            validation_loss, validation_accuracy = self.validation(x_validation,y_validation)

            if not epoch%print_every:
                print(f'epoch:{epoch},'+
                      f'acc:{accuracy:},',
                      f'loss:{loss},'
                      f'validation_acc:{validation_accuracy},'
                      f'validation_loss:{validation_loss},')

    def backward(self,output,y):
        self.softmax_classifier_output.backward(output,y)
        self.layers[-1].dinputs = self.softmax_classifier_output.dinputs

        for layer in reversed(self.layers[:-1]):
            layer.backward(layer.next.dinputs)

    def print_results(self):
        plt.plot(self.accuracy.epoch_accuracy)
        plt.xlabel("Accuracy")
        plt.ylabel("Epoch")
        plt.legend("Accuracy per Epoch")
        plt.show()

        plt.plot(self.loss.epoch_loss)
        plt.xlabel("Loss")
        plt.ylabel("Epoch")
        plt.legend("Loss per Epoch")
        plt.show()


class Layer_Input():
    def forward(self,inputs):
        self.output = inputs
