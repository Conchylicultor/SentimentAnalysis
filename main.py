#!/usr/bin/env python3

"""
Main script

Use python 3
"""

import os, sys
import random
import numpy as np
import params
import utils
import vocabulary
import rntnmodel

# Parametters
nbEpoch = 30
miniBatchSize = 1 # TODO = 30


def main():
    print("Welcome into RNTN implementation 0.1")
    
    random.seed("MetaMind") # Lucky seed ? Fixed seed for replication
    
    print("Parsing dataset, creating dictionary...")
    # Dictionary initialisation
    vocabulary.initVocab()
    
    # Loading dataset
    trainingSet = utils.loadDataset("trees/train.txt");
    print("Training loaded !")
    testingSet = utils.loadDataset("trees/test.txt");
    print("Testing loaded !")
    validationSet = utils.loadDataset("trees/dev.txt");
    print("Validation loaded !")
    
    print("Datasets loaded !")
    print("Nb of words", vocabulary.vocab.length());
    
    # Datatransform (normalisation, remove outliers,...) ?? > Not here
    
    # Creating the model
    model = rntnmodel.Model()
    
    # Plot the initial error (wait less time before seing if our model is learning)
    print("Initial errors...")
    trError = model.computeError(trainingSet, True)
    print("Train error: ", trError)
    teError = model.computeError(testingSet, True)
    print("Test  error: ", teError)
    
    print("Start training...")
    # TODO: Include the training in the cross-validation loop (tune parametters)
    # Main loop
    for i in range(nbEpoch):
        print("Epoch: ", i)
        
        # Randomly shuffle the dataset
        random.shuffle(trainingSet)
        
        # Loop over the training samples
        # TODO: Use mini-batch instead of online learning
        nbSampleCovered = 1 # To plot the progression of the epoch
        for trainingSample in trainingSet: # Select next the training sample
            # Forward pass
            model.evaluateSample(trainingSample) # Compute the output recursivelly
            
            # Backward pass (Compute the gradients for the current sample)
            gradient = model.backpropagate(trainingSample)
            
            # Add regularisation (the factor 2 will be multiplied < is useful for gradient checking)
            gradient = model.addRegularisation(gradient, miniBatchSize)
            
            # Update the weights
            model.updateWeights(gradient)
            
            # Plot progress every 10% of dataset covered
            if nbSampleCovered % (len(trainingSet)//10) == 0:
                print(nbSampleCovered*100 // len(trainingSet) + 1, "% of dataset covered")
            nbSampleCovered += 1
        
        # Compute new testing error
        print("Compute errors...")
        trError = model.computeError(trainingSet)
        print("Train error: ", trError)
        teError = model.computeError(testingSet, True)
        print("Test  error: ", teError)
        
        # Saving the model (every X epoch)
        print("Saving model...")
        model.saveModel("save/train") # Also save the dictionary
        
    print("Training complete, validating...")
    vaError = utils.computeError(validationSet, True)
    print("Validation error: ", vaError)
    

if __name__ == "__main__":
    main()
