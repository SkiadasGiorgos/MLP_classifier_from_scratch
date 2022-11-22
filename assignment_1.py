from keras.datasets import mnist
from layers import Layer
from activation_functions import Softmax, ReLU
from loss_functions import Loss, Softmax_loss_combination, CategoricalCrossEntropy
from optimization_functions import RMSProp_optimizer, SGD_optimization_mommentum, SGD_optimization
from accuracy_fucntion import Accuracy
from data_preprocessing import one_hot, shuffle_inputs, min_max_scaling
from models import Model

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

classes = 10
y_train = one_hot(y_train, 10)
y_test = one_hot(y_test, 10)
x_train = min_max_scaling(x_train)
x_test = min_max_scaling(x_test)

soft = Softmax()
optimizer = RMSProp_optimizer(learning_rate=1e-3, rho=0.99, decay=False, decay_rate=1e-2)
# optimizer = SGD_optimization_mommentum(learning_rate=1e-3, beta=.01)
accuracy = Accuracy()

soft_loss_combo = Softmax_loss_combination()
loss = CategoricalCrossEntropy

epochs = 100
model = Model()
model.add(Layer(784, 256, glorot=False,weight_regularizer=5e-4,bias_regularizer=5e-4))
model.add(ReLU())
model.add(Layer(256, 128, glorot=False,weight_regularizer=5e-4,bias_regularizer=5e-4))
model.add(ReLU())
model.add(Layer(128, 10, glorot=False,weight_regularizer=0,bias_regularizer=0))
model.add(Softmax())

model.set(loss=CategoricalCrossEntropy(),optimizer=optimizer,accuracy=accuracy)
model.finalize_model()
model.train(x_train,y_train,epochs=epochs,print_every=1,batch_size=128)
