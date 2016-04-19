#!/usr/bin/env python3

"""
Test the validation set
"""

import os, sys
import numpy as np
import utils
import rntnmodel
import vocabulary

inputModel = "save/train"

def main():
    print("Welcome into RNTN implementation 0.1")
    
    print("Loading dictionary...")
    # Dictionary initialisation
    vocabulary.initVocab(inputModel)
    
    # Loading dataset
    validationSet = utils.loadDataset("trees/dev.txt");
    print("Validation loaded !")
    
    # Creating the model
    model = rntnmodel.Model(inputModel)
            
    print("Computation validation...")
    vaError = model.computeError(validationSet, True)
    print("Validation error: ", vaError)
    

if __name__ == "__main__":
    main()
