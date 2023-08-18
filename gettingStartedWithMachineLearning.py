
import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

## Creates a neural network with one neuron and one layer
## Dense means that every neuron is connected to the neurons in the next layer
model = Sequential([Dense(units=1, input_shape=[1])])

## sgd ('stochastic graident of descent') is the optimizer for this network [https://www.geeksforgeeks.org/ml-stochastic-gradient-descent-sgd/]
## loss is the function used to calculate the accuracy of the predictions to the truth values in this case mean_squared_error
model.compile(optimizer='sgd', loss='mean_squared_error')

## formating the data and labels into a numpy array for tensorflow to use
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

## fits the data onto the labels using the model created on line 9
model.fit(xs, ys, epochs=500)

#prints a prediction on the model with an input of 10.0 (result is close to 19)
print(model.predict([10.0]))