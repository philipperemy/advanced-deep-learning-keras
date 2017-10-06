from time import sleep

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

x = tf.placeholder(tf.float32, shape=(1, 10))
y = tf.placeholder(tf.float32, shape=(1, 1))

y_ = 2 * slim.fully_connected(inputs=x, num_outputs=10, activation_fn=None)
y_ = 2 * slim.fully_connected(inputs=y_, num_outputs=1, activation_fn=None)

loss = tf.reduce_mean(tf.square(tf.subtract(y, y_)))
learning_rate = 0.001

tvs = tf.trainable_variables()
w_1 = tvs[0]
w_2 = tvs[2]

new_w_1 = tf.placeholder(tf.float32, shape=(10, 10))
assign_1_op = tf.assign(w_1, new_w_1)

new_w_2 = tf.placeholder(tf.float32, shape=(10, 1))
assign_2_op = tf.assign(w_2, new_w_2)


grads_1 = tf.gradients(loss, w_1)[0]
grads_2 = tf.gradients(loss, w_2)[0]

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

x_input = np.ones(shape=(1, 10,))
y_input = np.ones(shape=(1, 1))

print(tvs)
feed_dict = {x: x_input, y: y_input}

for i in range(1000):
    print(sess.run([y_, y, tvs], feed_dict))

    grad_1_values = sess.run(grads_1, feed_dict)
    print(grad_1_values)
    cur_w_1_values = sess.run(w_1, feed_dict)
    new_w_1_values = cur_w_1_values - learning_rate * grad_1_values
    sess.run(assign_1_op, {new_w_1: new_w_1_values})

    grad_2_values = sess.run(grads_2, feed_dict)
    print(grad_2_values)
    cur_w_2_values = sess.run(w_2, feed_dict)
    new_w_2_values = cur_w_2_values - learning_rate * np.expand_dims(grad_2_values.flatten(), axis=1)
    sess.run(assign_2_op, {new_w_2: new_w_2_values})

    print('loss = {}'.format(sess.run(loss, feed_dict)))
    sleep(0.5)

