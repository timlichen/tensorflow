# tf.contrib.learn => high-level TensorFlow library that simplifies the mechanics of machine learning, including the following, [running traning loops, running evaluation loops. managing data sets, managing feeding]

# tf.contrib.learn defines many common models.

import numpy as np
import tensorflow as tf
# declare list of features, we only have on real-valued feature
def model(features, labels, mode):
    # build a linear model and predict values
    W = tf.get_variable("W", [1], dtype=tf.float64)
    b = tf.get_variable("b", [1], dtype=tf.float64)
    y = W * features['x'] + b
    # loss sub-graph
    loss = tf.reduce_sum(tf.square(y-labels))
    # training sub-graph
    global_step = tf.train.get_global_step()
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = tf.group(optimizer.minimize(loss),
                     tf.assign_add(global_step, 1))
    # ModelFnOps connects subgraphs we built to the appropriate functionality.
    return tf.contrib.learn.ModelFnOps(
        mode=mode, predictions=y, loss=loss, train_op=train
        )


estimator = tf.contrib.learn.Estimator(model_fn=model)

# define our data sets

x = np.array([1.,2.,3.,4.])
y = np.array([0.,-1.,-2.,-3.])

input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x}, y, 4, num_epochs=1000)

# train
estimator.fit(input_fn=input_fn, steps=1000)

# evaluate our model
print(estimator.evaluate(input_fn=input_fn, steps=10))
