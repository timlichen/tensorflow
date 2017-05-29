 # Modified National Institute of Standards and Technology

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Implementing The Regression
# If you want to assign probabilities to an object being one of several different things, softmax is the thing to do.

# y = softmax(W[sub x] + b)

import tensorflow as tf
# Symbolic variable
x = tf.placeholder(tf.float32, [None, 784])
# Model Parameters
W = tf.Variable(tf.zeros([784, 10])) # tensors full of zeros
b = tf.Variable(tf.zeros([10]))

# W has a shape of [784, 10] because we want to multiply the 784-dimensional image vectors by it to produce 10-dimensional vectors of evidence for the difference classes. b has a shape of [10] so we can add it to the output.

y = tf.nn.softmax(tf.matmul(x, W) + b)

# 1. Multiply x by W with the expression tf.matmul(x, W).
# 2. Then add b
# 3. Apply tf.nn.softmax

# Training
