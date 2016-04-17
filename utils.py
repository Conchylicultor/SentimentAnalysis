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

def softClas(Ws, a):
    """
    Return the softmax sentiment prediction for the given word vector
    """
    return softmax(np.dot(Ws, a))

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

def computeError(dataset, V, W, Ws, regularisationTerm, compute = False):
    """
    Evaluate the cost error of the given dataset using the parametters
    Args:
        compute: If false, the dataset must have completed the forward pass with the given parametters
        before calling this function (the output will not be computed in this fct but the old one will 
        be used)
    Return:
        TODO: Return also the % of correctly classified labels (and the number) (by node ?? or just the root ?? < Both)
        In the paper, they uses 4 metrics (+/- or fine grained ; all or just root)
    """
    costRegularisation = regularisationTerm * (np.sum(W*W) + np.sum(Ws*Ws) + np.sum(V*V)) # TODO: Correct way to compute matrix multiplication (Check array vs matrix, norm, what about L ??...)
    
    # Evaluate error for each given sample
    costError = 0
    for sample in dataset:
        if compute: # If not done yet, compute the Rntn
            sample.computeRntn(V, W)
        costError += sample.evaluateCost(Ws) # Normalize also by number of nodes ??
    costError /= len(dataset) # Normalize the cost by the number of sample
    costError += costRegularisation # Add regularisation (add N times, then normalized)
    
    return costError
