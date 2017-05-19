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
