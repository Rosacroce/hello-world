# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 20:10:16 2020

Creating a simple Neural Network from scratch.
The network is made up of the following essential parts, listed in sequencial order from input to output:
    - inputs
    - synapses (weights)
    - a neuron
    - output

It should be noted that there are no hidden layers.
This elementary network architecture is also called 'perceptron'.

The example on which the network is based is a very basic binary classification problem (see Source)
In the example the input is made up of 3 features x1, x2 and x3. The output is 1 or 0.

Source:
    https://www.youtube.com/watch?v=kft1AJ9WVDk

@author: aless
"""

import numpy as np

# NORMALISING FUNCTION
def sigmoid(x):
    ''' sigmoid activation function '''
    return 1 / (1 + np.exp(-x))     # it's value between 0 and 1

# ADJUSTMENT FUNCTION
def sigmoid_derivative(x):
    ''' derivative of sigmoid function used to calculate the adjustment to the weights in the training phase '''
    return x * (1 - x)


# TRAINING VALUES
# input
training_inputs = np.array([[0,0,1],    # [x1, x2, x3] are the 3 features of every input value
                            [1,1,1],
                            [1,0,1],
                            [0,1,1]])

# output
training_outputs = np.array([[0,1,1,0]]).T

# WEIGHTS INITIALISATION
# weights are initialised using random values
np.random.seed(1)   #
synaptic_weights = 2 * np.random.random((3, 1)) - 1

print(f'Random starting synaptic weights: {synaptic_weights}')

# MAIN

train_iterations = 20000         # number of iteration for training. Every iteration the weights are adjusted

for iteration in range(train_iterations):
    input_layer = training_inputs

    outputs = sigmoid(np.dot(input_layer, synaptic_weights))    # matrix multiplication inputs x weights

    error = training_outputs - outputs                          # utilised to derive the adjustment to the weights

    adjustements = error * sigmoid_derivative(outputs)

    synaptic_weights += np.dot(input_layer.T, adjustements)

print(f'weights after training {synaptic_weights}')

print(f'outputs after training: {outputs}')






