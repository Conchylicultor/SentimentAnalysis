#!/usr/bin/env python3

"""
Some utilities functions
"""

import numpy as np

def softmax(w):
    """
    Straightforward implementation of softmax function
    """
    e = np.exp(np.array(w))
    dist = e / np.sum(e)
    return dist
