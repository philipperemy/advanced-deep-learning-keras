import keras
import numpy as np
from keras.datasets import mnist
from keras.layers import Dense, Dropout
from keras.layers.core import RepeatVector
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential

batch_size = 128
num_classes = 10
epochs = 20

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

vocab_size = 26

y_train = np.reshape(np.tile(y_train, (vocab_size,)), (-1, 10, vocab_size))
y_test = np.reshape(np.tile(y_test, (vocab_size,)), (-1, 10, vocab_size))

m = Sequential()
m.add(Dense(512, activation='relu', input_shape=(784,)))
m.add(Dropout(0.2))
m.add(Dense(512, activation='relu'))
m.add(Dropout(0.2))
m.add(RepeatVector(10))
m.add(LSTM(11, return_sequences=True))
m.add(TimeDistributed(Dense(vocab_size, activation='softmax')))

m.summary()
m.compile(loss='categorical_crossentropy',
          optimizer='adam',
          metrics=['accuracy'])

history = m.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=(x_test, y_test))
pred = m.predict(x_test)
print(pred)
