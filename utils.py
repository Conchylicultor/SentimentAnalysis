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
    e = np.exp(np.array(w))
    dist = e / np.sum(e)
    return dist

def actFct(x):
    """
    The NN activation function (here tanh)
    """
    return np.tanh(x)

def actFctDer(x):
    """
    Derivate of the activation function
    """
    return 1.0 - np.tanh(x)**2

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

def computeError(dataset, Ws, V, W, regularisationTerm):
    """
    Evaluate the cost error of the given dataset using the parametters
    WARNING: the dataset must have completed the forward pass with the given parametters
    before calling this function (the output is not computed in this error)
    """
    costRegularisation = regularisationTerm * (norm(W*W) + norm(Ws*Ws) + norm(V*V)) # TODO: Correct way to compute matrix multiplication (array vs matrix, norm, what about L ??...)
    
    # Evaluate error for each given sample
    costError = 0
    for sample in dataset:
        costError += sample.evaluateCost() + costRegularisation
    costError /= len(dataset) # Normalize the cost
    
    return costError
