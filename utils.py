#!/usr/bin/env python3

"""
Some utilities functions
"""

import numpy as np
import tree

def softmax(w):
    """
    Straightforward implementation of softmax function
    """
    e = np.exp(np.array(w))
    dist = e / np.sum(e)
    return dist

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
