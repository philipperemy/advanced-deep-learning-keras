import numpy as np
from keras import regularizers
from keras.callbacks import Callback
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

# Just show how the weights are distributed.
# L1 => only one should be big.
# L2 => all should be fairly similar and small.

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)


class My_Callback(Callback):
    def on_epoch_begin(self, epochs, logs={}):
        a = self.model.get_weights()
        print(np.concatenate(a).flatten())


batch_size = 128
num_classes = 10
epochs = 20

reg_1 = Dense(1, activation=None, input_shape=(10,), kernel_regularizer=regularizers.l2(0.01), use_bias=False)
reg_2 = Dense(1, activation=None, input_shape=(10,), kernel_regularizer=regularizers.l1(0.1), use_bias=False)
no_reg = Dense(1, activation=None, input_shape=(10,), use_bias=False)

model = Sequential()
model.add(reg_2)

adam = Adam(lr=0.0001)
model.summary()
model.compile(loss='mse', optimizer=adam)

my_callback = My_Callback()

size = 1000
val = np.random.standard_normal(size=size)
x = np.reshape(np.repeat(val, 10, axis=0), (-1, 10))
y = np.mean(x, axis=1)
model.fit(x, y, epochs=100000, verbose=1, callbacks=[my_callback])  # 400
model.fit(x, y, epochs=5, verbose=1)

print(np.sum(np.concatenate(model.get_weights()).flatten()))
print(model.get_weights())

print(model.predict(np.ones((1, 10)), batch_size=1))
