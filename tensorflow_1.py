'''
TENSOR

3 # a rank 0 tensor; this is a scalar with shape []
[1. ,2., 3.] # a rank 1 tensor; this is a vector with shape [3]
[[1., 2., 3.], [4., 5., 6.]] # a rank 2 tensor; a matrix with shape [2, 3]
[[[1., 2., 3.]], [[7., 8., 9.]]] # a rank 3 tensor with shape [2, 1, 3]
'''

import tensorflow as tf

# Two discrete sections
#   1. Computational Graph
#   2. Running the Computational Graph

# each node takes 0 or more tensors as inputs and produces a tensor as an output

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) # implicit tf.float32

# print(dir(node1))

print(node1, node2)

# when evaluated, node1 and node2 will produce 3.0 and 4.0 respectively. To evaluate the nodes, we must run the computational graph within a session, which encapsulates the control and state of the TF runtime.

# create session obj.
sess = tf.Session()
print(sess.run([node1, node2]))

node3 = tf.add(node1, node2)
print('node3:', node3)
print('sess.run(node3):', sess.run(node3))

# until now graphs have been dull, results are constants, but we can create parameterized graphs that accept external inputs, known as placeholders. A placeholder is a promise to prive a value later.

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder_node = a + b # + provides a shortcut fot tf.add(a, b)

# evalute the graph with multiple inputs by using the feed_dict parameter to specify Tensors that provide concrete value to these placehodlers.

print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))

# adding another operation...

add_and_triple = adder_node * 3
print(sess.run(add_and_triple, {a: 3, b: 4.5}))

# typically in ML, we will want a model that can take arbitrary inputs, like above. To make the model trainable, we need to be able to modify the graph to get new outputs with the same inputs. Variables allows us to add trainable parameters to a graph. They are constructed with a type and initial value.

W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)

linear_model = W * x + b

# constants are initialized when tf.constant is called, but tf.Variable needs special methods to initialize. You must explicity call a special operation.

init = tf.global_variables_initializer()
sess.run(init) # init is a handle to the TensorFlow sub-graph that inits all the global variables. Until we call sess.run, the variables are uninitialized.

print(sess.run(linear_model, {x:[1,2,3,4]}))
#=> [ 0.          0.30000001  0.60000002  0.90000004]

'''
Not bad, but we don't know how good it is yet. To evaluate the model on the traning data, we need a y placeholder to provide the desired values, and we need to write a loss function.

wtf is a loss function? It's a function that measures how far apart the current model is from the provided data. We'll use a standard loss model for linear regression, which sums the squares of the deltas between the current model and the provided data.

linear_model - y

create a vector where each element is the corresponding example's error delta. We call tf.square to square that error. Then we sum all the squared errors to create a single scalar that abstracts the error of all examples using tf.reduce_sum:
'''


y = tf.placeholder(tf.float32)

deltas = linear_model - y

print(sess.run(deltas, {x:[1,2,3,4], y: [0,-1,-2,-3]}))
# [ 0.          1.29999995  2.5999999   3.9000001 ]
# Taking each element of the linear model list and subtracting the correlating indexs element from y (the provided data) and populating the list, eg.
# [ 0 - 0, 0.3-(-1), .6-(-2), ... ]

squared_deltas = tf.square(linear_model - y)
print(sess.run(squared_deltas, {x:[1,2,3,4], y: [0,-1,-2,-3]}))
# [  0.           1.68999982   6.75999928  15.21000099]

loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x:[1,2,3,4], y: [0,-1,-2,-3]}))
#  23.66
