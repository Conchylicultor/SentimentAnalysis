#!/usr/bin/env python3

"""
Some utilities functions
"""

import numpy as np
import tree

# Maths functions

def softmax(w):
    """
    Straightforward implementation of softmax function
    """
    e = np.exp(w)
    dist = e / np.sum(e)
    return dist

def actFct(x):
    """
    The NN activation function (here tanh)
    """
    return np.tanh(x)

def actFctDerFromOutput(x):
    """
    Derivate of the activation function
    WARNING: In this version, we take as input an output value 
    after the activation function (x = tanh(output of the tensor)).
    This works because the derivate of tanh is function of tanh
    """
    return 1.0 - x**2

#def actFctDer(x):
    #"""
    #Derivate of the activation function
    #"""
    #return 1.0 - np.tanh(x)**2

# Other utils functions
    
def loadDataset(filename):
    """
    Load and return the dataset given in parametter
    """

    dataFile = open(filename, 'r')
    lines = dataFile.readlines()
    dataFile.close()

    dataset = []

    # Extract rows
    for line in lines:
        dataset.append(tree.Tree(line)) # Create the tree for each sentence

    return dataset

