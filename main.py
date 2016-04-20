#!/usr/bin/env python3

"""
Main script

Use python 3
"""

import os, sys
import random
import numpy as np
import train
import utils
import vocabulary

# Parametters
nbEpoch = 30
miniBatchSize = 25
adagradResetNbIter = 5 # Reset every X iterations (0 for never)

learningRate = 0.05
regularisationTerm = 0.0001

# Path and name where the infos will be saved
outputDir = "save/"
outputNameDefault = "train"

def main(outputName):
    print("Welcome into RNTN implementation 0.6 (recording will be on ", outputName, ")")
    
    random.seed("MetaMind") # Lucky seed ? Fixed seed for replication
    np.random.seed(7)
    
    print("Parsing dataset, creating dictionary...")
    # Dictionary initialisation
    vocabulary.initVocab()
    
    # Loading dataset
    datasets = {}
    datasets['training'] = utils.loadDataset("trees/train.txt");
    print("Training loaded !")
    datasets['testing'] = utils.loadDataset("trees/test.txt");
    print("Testing loaded !")
    datasets['validating'] = utils.loadDataset("trees/dev.txt");
    print("Validation loaded !")
    
    print("Datasets loaded !")
    print("Nb of words", vocabulary.vocab.length());
    
    # Datatransform (normalisation, remove outliers,...) ?? > Not here
    
    # Cross-validation loop (too long for complete k-fold cross validation so just train/test)
    for miniB in [25, 1]: # MiniBatchSize
        for resetAda in [0, 3, 6, 10]: # Adagrad reset
            for learnRate in [0.1, 0.01, 0.001]: # Learning rate
                for regular in [0, 0.00001, 0.0001, 0.001]: # Regularisation
                    params = {}
                    params["nbEpoch"]            = nbEpoch
                    params["learningRate"]       = learnRate
                    params["regularisationTerm"] = regular
                    params["adagradResetNbIter"] = resetAda
                    params["miniBatchSize"]      = miniB
                    # No need to reset the vocabulary values (contained in model.L so automatically reset)
                    # Same for the training and testing set (output values recomputed at each iterations)
                    model, error = train.train(outputName, datasets, params)

    # TODO: Plot the cross-validation curve

    ## Not here
    #print("Training complete, validating...")
    #vaError = model.computeError(datasets['validating'], True)
    #print("Validation error: ", vaError)

    print("The End. Thank you for using this program!")
    

if __name__ == "__main__":
    # Simple parsing to get the model name
    outputName = outputNameDefault
    if len(sys.argv) > 1:
        outputName = sys.argv[1]
    main(outputDir + outputName)
