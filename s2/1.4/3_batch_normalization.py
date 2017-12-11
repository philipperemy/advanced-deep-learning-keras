import numpy as np
from keras.callbacks import Callback
from keras.layers import Dense
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)


class My_Callback(Callback):
    def on_epoch_begin(self, epochs, logs={}):
        a = self.model.get_weights()
        np.linalg.norm(a[0])
        print(np.linalg.norm(a[0]))
        # print(np.concatenate(a).flatten())


# instantiate model
model = Sequential()

# we can think of this chunk as the input layer
model.add(Dense(64, input_shape=(10,), init='uniform'))
model.add(BatchNormalization())
model.add(Dense(1, activation=None))

# setting up the optimization of our weights
model.compile(loss='mse', optimizer='adam')

BIG_NUMBER = 1000
X_train = np.ones(shape=(1, 10))
y_train = np.ones(shape=(1, 1)) * BIG_NUMBER

# explicitly give small input and enormous output to force the weights to become really big.
# if the weights become really big, because the input is very small, the activations will become very big.

# callback = My_Callback()

# running the fitting
# model.load_weights('weights.h5')
model.fit(X_train, y_train, epochs=5000)
# model.save_weights('weights.h5')

from read_activations import get_activations
a = get_activations(model, X_train)
