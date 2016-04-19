#!/usr/bin/env python3

"""
Main script

Use python 3
"""

import os, sys
import random
import numpy as np
import utils
import vocabulary
import rntnmodel

# Parametters
nbEpoch = 30
miniBatchSize = 1 # TODO Try with 30
adagradResetNbIter = 5 # Reset every X iterations

# Path and name where the infos will be saved
outputModel = "save/model"
outputResult = "results/train.csv"

def main():
    print("Welcome into RNTN implementation 0.6")
    
    random.seed("MetaMind") # Lucky seed ? Fixed seed for replication
    np.random.seed(7)
    
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
    #print("Initial errors...")
    #trError = model.computeError(trainingSet, True)
    #print("Train error: ", trError)
    #teError = model.computeError(testingSet, True)
    #print("Test  error: ", teError)
    
    print("Start training...")
    # TODO: Include the training in the cross-validation loop (tune parametters)
    
    # Indicate a new training on the result file
    resultFile = open(outputResult, "a") # Open the file (cursor at the end)
    resultFile.write("Epoch|Train|Test\n") # Record the data for the learning curve
    resultFile.close()
    
    # Main loop
    for i in range(nbEpoch):
        print("Epoch: ", i)
        
        # Randomly shuffle the dataset
        random.shuffle(trainingSet)
        
        if i % adagradResetNbIter == 0: # Reset every 4
            model.resetAdagrad() # Start with a clear history
        
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
                print("%d%% of dataset covered (%d/%d)" % ((nbSampleCovered*100 // len(trainingSet) + 1), nbSampleCovered, len(trainingSet)))
            nbSampleCovered += 1
        
        # Compute new testing error
        print("Compute errors...")
        trError = model.computeError(trainingSet)
        print("Train error: ", trError)
        teError = model.computeError(testingSet, True)
        print("Test  error: ", teError)
        
        # Saving the model (for each epoch)
        print("Saving model...")
        model.saveModel(outputModel) # The function also save the dictionary
        
        resultFile = open(outputResult, "a") # Open the file (cursor at the end)
        resultFile.write("%d|%s|%s\n" % (i, trError.toCsv(), teError.toCsv())) # Record the data for the learning curve
        resultFile.close()
        
        
    print("Training complete, validating...")
    vaError = model.computeError(validationSet, True)
    print("Validation error: ", vaError)
    
    print("The End. Thank you for using this program!")
    

if __name__ == "__main__":
    main()
