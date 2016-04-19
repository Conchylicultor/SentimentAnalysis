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
miniBatchSize = 25
adagradResetNbIter = 0 # Reset every X iterations (0 for never)

# Path and name where the infos will be saved
outputDir = "save/"
outputNameDefault = "training"

def main(outputModel):
    print("Welcome into RNTN implementation 0.6 (recording will be on ", outputModel, ")")
    
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
    model = rntnmodel.Model(regularisationTerm = 0)
    
    # Plot the initial error (wait less time before seing if our model is learning)
    #print("Initial errors...")
    #trError = model.computeError(trainingSet, True)
    #print("Train error: ", trError)
    #teError = model.computeError(testingSet, True)
    #print("Test  error: ", teError)
    
    # TODO: Include the training in the cross-validation loop (tune parametters)
    print("Start training...")
    print("Parametters:")
    paramsStr  = "- Minibatch size: %d\n" % miniBatchSize
    paramsStr += "- Learning rate: %f\n" % model.learningRate
    paramsStr += "- Regularisation: %f\n" % model.regularisationTerm
    paramsStr += "- Adagrad reset after: %d\n" % adagradResetNbIter
    print(paramsStr)
    
    # Indicate a new training on the result file
    resultFile = open(outputModel + "_train.csv", "a") # Open the file (cursor at the end)
    resultFile.write(paramsStr)
    resultFile.write("Epoch|Train|Test\n") # Record the data for the learning curve
    resultFile.close()
    
    # Main loop
    for i in range(nbEpoch):
        print("Epoch: ", i)
        
        # Randomly shuffle the dataset
        random.shuffle(trainingSet)
        
        if adagradResetNbIter != 0 and i % adagradResetNbIter == 0: # Reset every X epochs
            model.resetAdagrad() # Start with a clear history
        
        # Loop over the training samples
        nbSampleCovered = 1 # To plot the progression of the epoch
        gradient = None
        currentBatch = 0
        for trainingSample in trainingSet: # Select next the training sample
            # Forward pass
            model.evaluateSample(trainingSample) # Compute the output recursivelly
            
            # Backward pass (Compute the gradients for the current sample)
            if gradient is None:
                gradient = model.buildEmptyGradient() # Initialize the gradient
            
            gradient += model.backpropagate(trainingSample)
            
            # Minibatch: add the gradient only after X samples
            currentBatch += 1
            if currentBatch >= miniBatchSize:
                # Add regularisation (the factor 2 will be multiplied < is useful for gradient checking)
                gradient = model.addRegularisation(gradient, miniBatchSize) # Average the gradient over the miniBatchSize
                # Update the weights
                model.updateWeights(gradient)
                # Reset current batch and gradient
                currentBatch = 0
                gradient = None
            
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
        
        resultFile = open(outputModel + "_train.csv", "a") # Open the file (cursor at the end)
        resultFile.write("%d|%s|%s\n" % (i, trError.toCsv(), teError.toCsv())) # Record the data for the learning curves
        resultFile.close()
        
        
    print("Training complete, validating...")
    vaError = model.computeError(validationSet, True)
    print("Validation error: ", vaError)
    
    print("The End. Thank you for using this program!")
    

if __name__ == "__main__":
    # Simple parsing to get the model name
    outputName = outputNameDefault
    if len(sys.argv) > 1:
        outputName = sys.argv[1]
    main(outputDir + outputName)
