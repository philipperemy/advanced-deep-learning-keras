import numpy as np
import tensorflow as tf

import tensorflow.contrib.slim as slim

x = tf.placeholder(tf.float32, shape=(1, 10,))
y = tf.placeholder(tf.float32, shape=(1, 1))

# MODEL IS: Y = 2 * W * X
# DERIVATIVE OF Y REGARDING TO X is 2 * W. DY/DX = 2 * W.

y_ = 2 * slim.fully_connected(inputs=x, num_outputs=1, activation_fn=None)

grads = tf.gradients(y_, x)[0]

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

gradient_values = sess.run(grads, {x: np.ones(shape=(1, 10,)),
                                   y: np.ones(shape=(1, 1))}).flatten()

print('grads = ', gradient_values)

weight_values = sess.run(tf.trainable_variables(), {x: np.ones(shape=(1, 10,)),
                                                    y: np.ones(shape=(1, 1))})[0].flatten()

print('weights =', weight_values)

np.testing.assert_almost_equal(gradient_values.flatten(), 2 * weight_values)
