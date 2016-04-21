#!/usr/bin/env python3

"""
Train the model with the given params
"""

import os, sys
import random
import numpy as np
import matplotlib.pyplot as plt
import rntnmodel

def train(outputName, datasets, params):
    """
    Train the model on the given arguments, record the results
    Args:
        outputName: the model and results will be saved under this name (string)
        datasets: dictionary containing the testing, training (no validation set)
        params: dictionary containing the parameters Names/Values
    Return:
        model: the trained model
        error: an error object containing the different metrics for the testing set
    """
    
    print("Start training...")
    
    # Plot the parameters
    print("Parameters:")
    for key in sorted(params): # Get all params
        print("-", key, ":", params[key])
        outputName += "-" + key + "=" + str(params[key])

    # Extract the given arguments
    trainingSet = datasets["training"]
    testingSet  = datasets["testing"]
    
    assert len(params) == 5 # In case we will need to change or add a parameters someday, make sure all parameters are extracted and used
    nbEpoch            = params["nbEpoch"]
    learningRate       = params["learningRate"]
    regularisationTerm = params["regularisationTerm"]
    adagradResetNbIter = params["adagradResetNbIter"]
    miniBatchSize      = params["miniBatchSize"]
    
    # Creating the model
    model = rntnmodel.Model(
        regularisationTerm = regularisationTerm,
        learningRate = learningRate
        )
    
    trErrors = []
    teErrors = []
    # Plot the initial error (wait less time before seing if our model is learning)
    #print("Initial errors...")
    #trError = model.computeError(trainingSet, True)
    #print("Train error: ", trError)
    #teError = model.computeError(testingSet, True)
    #print("Test  error: ", teError)
    #trErrors.append(trError)
    #teErrors.append(teError)
    
    # Indicate a new training on the result file
    resultFile = open(outputName + "_train.csv", "a") # Open the file (cursor at the end to not erasing eventual previous results)
    resultFile.write("Epoch|TrainCost|TrainAll|TrainRoot|TestCost|TestAll|TestRoot\n") # Record the data for the learning curve (format)
    resultFile.close()
    
    # Main loop
    for i in range(nbEpoch):
        print("Epoch: %d/%d" % (i+1, nbEpoch))
        
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
                # Add regularisation (the factor 2 will be multiplied << is useful for gradient checking)
                gradient = model.addRegularisation(gradient, miniBatchSize) # Will average the gradient over the miniBatchSize
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
        
        trErrors.append(trError)
        teErrors.append(teError) # Keep track of the errors for the curves
        
        # Saving the model (at each epoch)
        print("Saving model...")
        model.saveModel(outputName) # The function also save the dictionary
        
        resultFile = open(outputName + "_train.csv", "a") # Open the file (cursor at the end)
        resultFile.write("%d|%s|%s\n" % (i, trError.toCsv(), teError.toCsv())) # Record the data for the learning curves
        resultFile.close()
        
    # Record the learning curve
    xEpoch = []
    trRegCost = []
    teRegCost = []
    trAllNodes = []
    teAllNodes = []
    trRoot = []
    teRoot = []
    for i in range(len(trErrors)): # Construct the datas
        xEpoch.append(i)
        trRegCost.append(trErrors[i].getRegCost())
        teRegCost.append(teErrors[i].getRegCost())
        trAllNodes.append(trErrors[i].getPercentNodes())
        teAllNodes.append(teErrors[i].getPercentNodes())
        trRoot.append(trErrors[i].getPercentRoot())
        teRoot.append(teErrors[i].getPercentRoot())
    
    # Create and save the graphs
    # TODO: Add labels 'Training'/'Testing' (even if kind of obvious)
    # TODO: Fixed axis ? (easier to compare)
    
    plt.figure(1, figsize=(20, 10), dpi=80)
    plt.clf() # Reset
    
    plt.subplot(2, 2, 1)
    plt.plot(xEpoch, trRegCost)
    plt.plot(xEpoch, teRegCost)
    plt.grid(True)
    plt.title('Cost')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    
    plt.subplot(2, 2, 3)
    plt.plot(xEpoch, trAllNodes)
    plt.plot(xEpoch, teAllNodes)
    plt.grid(True)
    plt.title('All nodes')
    plt.xlabel('Epoch')
    plt.ylabel('% of success')
    
    plt.subplot(2, 2, 4)
    plt.plot(xEpoch, trRoot)
    plt.plot(xEpoch, teRoot)
    plt.grid(True)
    plt.title('Root only')
    plt.xlabel('Epoch')
    plt.ylabel('% of success')
    
    plt.savefig(outputName + '_learningCurve.png')
    
    # Return perfs at the end
    errors = []
    return model, errors
