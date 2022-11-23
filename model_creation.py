from keras.datasets import mnist
from layers import Layer
from activation_functions import Softmax, ReLU
from loss_functions import Loss, Softmax_loss_combination, CategoricalCrossEntropy
from optimization_functions import RMSProp_optimizer, SGD_optimization_mommentum, SGD_optimization, Adam
from accuracy_fucntion import Accuracy
from data_preprocessing import one_hot, min_max_scaling, shuffle_inputs
from models import Model
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

classes = 10
y_train = one_hot(y_train, 10)
y_test = one_hot(y_test, 10)
x_train = min_max_scaling(x_train)
x_test = min_max_scaling(x_test)

x_train, y_train = shuffle_inputs(x_train, y_train)
x_validation = x_train[:int(.3 * x_train.shape[0])]
y_validation = y_train[:int(.3 * x_train.shape[0])]
x_train = x_train[int(.3 * y_train.shape[0]):]
y_train = y_train[int(.3 * y_train.shape[0]):]

soft = Softmax()
# optimizer = RMSProp_optimizer(learning_rate=1e-2, rho=0.99, decay=False, decay_rate=1e-3)
# optimizer = SGD_optimization_mommentum(learning_rate=1e-3, beta=.01)
# optimizer = SGD_optimization(learning_rate=1e-2)
optimizer = Adam(learning_rate=1e-3, decay=0, beta1=.9, beta2=0.99)
accuracy = Accuracy()

soft_loss_combo = Softmax_loss_combination()
loss = CategoricalCrossEntropy

epochs = 10
model = Model()
model.add(Layer(x_train.shape[1], 128, glorot=False, weight_regularizer=0, bias_regularizer=0))
model.add(ReLU())
model.add(Layer(128, 128, glorot=False, weight_regularizer=0, bias_regularizer=0))
model.add(ReLU())
model.add(Layer(128, 10, glorot=False, weight_regularizer=0, bias_regularizer=0))
model.add(Softmax())

model.set(loss=CategoricalCrossEntropy(), optimizer=optimizer, accuracy=accuracy)
model.finalize_model()
model.train(x_train, y_train, epochs=epochs, print_every=1, batch_size=128, x_validation=x_validation,
            y_validation=y_validation)
model.print_results()
model.validation(x_test,y_test, test=True)
