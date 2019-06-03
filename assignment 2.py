Python 3.6.7 (v3.6.7:6ec5cf24b7, Oct 20 2018, 13:35:33) [MSC v.1900 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> Python 3.6.7 (v3.6.7:6ec5cf24b7, Oct 20 2018, 13:35:33) [MSC v.1900 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> #import the necessary libraries first
... import numpy as np
>>> import tensorflow as tf
>>> import matplotlib.pyplot as plt
>>> np.random.seed(101)
>>> tf.set_random_seed(101)
>>> # Genrating random linear data
... # There will be 50 data points ranging from 0 to 50
... x = np.linspace(0, 50, 50)
>>> # Adding noise to the random linear data
... x += np.random.uniform(-4, 4, 50)
>>> y = np.linspace(0, 0, 50)
>>> y += np.random.normal(0, 1, 50)
>>> n = len(x) # Number of data points
>>> # Plot of Training Data
... plt.scatter(x, y)
<matplotlib.collections.PathCollection object at 0x000001E3A5737CC0>
>>> plt.xlabel('x')
Text(0.5, 0, 'x')
>>> plt.ylabel('y')
Text(0, 0.5, 'y')
>>> plt.title("Training Data")
Text(0.5, 1.0, 'Training Data')
>>> plt.show()
>>> X = tf.placeholder("float")
>>> Y = tf.placeholder("float")
>>> W = tf.Variable(np.random.randn(), name = "W")
WARNING:tensorflow:From C:\Users\Keerthi\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\framework\op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
Instructions for updating:
Colocations handled automatically by placer.
>>> b = tf.Variable(np.random.randn(), name = "b")
>>> learning_rate = 0.01
>>>
>>> training_epochs = 1000
>>> y_pred1 = tf.add(tf.multiply(X, W), b)
>>> cost = tf.reduce_sum(tf.pow(y_pred1-Y, 2)) / (2 * n)
>>> # Gradient Descent Optimizer
... optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
WARNING:tensorflow:From C:\Users\Keerthi\AppData\Local\Programs\Python\Python36\lib\site-packages\tensorflow\python\ops\math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.cast instead.
>>>
>>> # Global Variables Initializer
... init = tf.global_variables_initializer()
>>> # Starting the Tensorflow Session
... with tf.Session() as sess:
...     sess.run(init)
...     # Iterating through all the epochs
...     for epoch in range(training_epochs):
...         # Feeding each data point into the optimizer using Feed Dictionary
...         for (_x, _y) in zip(x, y):
...             sess.run(optimizer, feed_dict = {X : _x, Y : _y})
...         # Displaying the result after every 50 epochs
...         if (epoch + 1) % 50 == 0:
...             # Calculating the cost a every epoch
...             c = sess.run(cost, feed_dict = {X : x, Y : y})
...             print("Epoch", (epoch + 1), ": cost =", c, "W =", sess.run(W), "b =", sess.run(b))
...  # Storing necessary values to be used outside the Session
...     training_cost = sess.run(cost, feed_dict ={X: x, Y: y})
...     weight = sess.run(W)
...     bias = sess.run(b)
...
2019-06-03 19:54:05.675383: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
Epoch 50 : cost = 0.61677545 W = 0.022213679 b = -0.49482715
Epoch 100 : cost = 0.6048826 W = 0.020790085 b = -0.42521897
Epoch 150 : cost = 0.5956166 W = 0.019518534 b = -0.3630452
Epoch 200 : cost = 0.588423 W = 0.018382818 b = -0.30751306
Epoch 250 : cost = 0.58286124 W = 0.017368414 b = -0.2579128
Epoch 300 : cost = 0.5785825 W = 0.016462374 b = -0.213611
Epoch 350 : cost = 0.57531047 W = 0.015653118 b = -0.17404154
Epoch 400 : cost = 0.5728263 W = 0.014930313 b = -0.13869937
Epoch 450 : cost = 0.5709573 W = 0.014284714 b = -0.10713214
Epoch 500 : cost = 0.5695671 W = 0.013708079 b = -0.07893695
Epoch 550 : cost = 0.5685479 W = 0.013193037 b = -0.053753447
Epoch 600 : cost = 0.5678151 W = 0.012733014 b = -0.03126006
Epoch 650 : cost = 0.5673023 W = 0.01232213 b = -0.011169439
Epoch 700 : cost = 0.5669574 W = 0.011955135 b = 0.0067751193
Epoch 750 : cost = 0.5667394 W = 0.011627343 b = 0.022802811
Epoch 800 : cost = 0.56661665 W = 0.011334568 b = 0.037118427
Epoch 850 : cost = 0.5665644 W = 0.011073064 b = 0.049904883
Epoch 900 : cost = 0.56656355 W = 0.010839496 b = 0.061325517
Epoch 950 : cost = 0.56659925 W = 0.010630878 b = 0.07152609
Epoch 1000 : cost = 0.56666034 W = 0.010444543 b = 0.080637135
>>> # Calculating the predictions
... predictions = weight * x + bias
>>> print("Training cost =", training_cost, "Weight =", weight, "bias =", bias, '\n')
Training cost = 0.56666034 Weight = 0.010444543 bias = 0.080637135

>>> # Plotting the Results
... plt.plot(x, y, 'ro', label ='Original data')
[<matplotlib.lines.Line2D object at 0x000001E3A59E5208>]
>>> plt.plot(x, predictions, label ='Fitted line')
[<matplotlib.lines.Line2D object at 0x000001E3A5E82860>]
>>> plt.title('Linear Regression Result')
Text(0.5, 1.0, 'Linear Regression Result')
>>> plt.legend()
<matplotlib.legend.Legend object at 0x000001E3A59C6BE0>
>>> plt.show()
>>>
