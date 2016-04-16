#!/usr/bin/env python3

"""
Test the validation set
"""

import os, sys
import numpy as np
import utils
import vocabulary


def main():
    print("Welcome into RNTN implementation 0.1")
    
    print("Loading dictionary...")
    # Dictionary initialisation
    vocabulary.initVocab("save/dict")
    
    # Loading dataset
    validationSet = utils.loadDataset("trees/dev.txt");
    print("Validation loaded !")
    
    # Creating the model
    modelFile = np.load("save/model")
    V  = modelFile['V']  # Tensor of the RNTN layer
    W  = modelFile['W']  # Regular term of the RNTN layer
    Ws = modelFile['Ws'] # Softmax classifier
    #L = # Vocabulary (List of N words on vector representation) << Contained in the vocab variable
            
    print("Computation validation...")
    vaError = utils.computeError(validationSet,  V, W, Ws, regularisationTerm, True)
    print("Validation error: ", vaError)
    

if __name__ == "__main__":
    main()
