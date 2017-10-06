import numpy as np
import tensorflow as tf

# TENSORFLOW CODE

input_size = 3

x = tf.placeholder(tf.float32, shape=(1, input_size))

weights = tf.Variable(tf.random_normal((input_size, 1)), name="weights")
biases = tf.Variable(tf.zeros(shape=1), name="biases")

model = tf.nn.sigmoid(tf.matmul(x, weights) + biases)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

print(sess.run(model, feed_dict={x: np.random.uniform(size=(1, input_size))}))

# KERAS CODE

from keras.models import Sequential
from keras.layers import Dense

m = Sequential()
m.add(Dense(1, input_shape=(1, input_size), activation='sigmoid'))

m.compile(optimizer='adam', loss='mse')

print(m.predict(np.random.uniform(size=(1, 1, input_size)))[0])
