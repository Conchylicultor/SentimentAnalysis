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
learningRate = 0.1 # TODO: Replace by AdaGrad !!
miniBatchSize = 1 # TODO = 30
regularisationTerm = 0.0001 # Lambda

def main():
    print("Welcome into RNTN implementation 0.1")
    
    random.seed("MetaMind") # Fixed seed
    
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
            
            # Backward pass (Compute the gradients)
            gradientV, gradientW, gradientWs = trainingSample.backpropagateRntn(V, W, Ws)
            
            # Add regularisation (we don't care about the factor 2 contained in regularisationTerm ? < could be useful for gradient checking)
            gradientV  += regularisationTerm * miniBatchSize * V
            gradientW  += regularisationTerm * miniBatchSize * W
            gradientWs += regularisationTerm * miniBatchSize * Ws
            # What about L ??
            
            # Update the weights
            V  -= learningRate * gradientV # Step in the oposite direction of the gradient
            W  -= learningRate * gradientW
            Ws -= learningRate * gradientWs
            # L is updated when calling backpropagateRntn
            
            # Plot progress every 10% of dataset covered
            if nbSampleCovered % (len(trainingSet)//10) == 0:
                print(nbSampleCovered*100 // len(trainingSet), "% of dataset covered")
            nbSampleCovered += 1
        
        # Compute new testing error
        print("Compute errors...")
        trError = utils.computeError(trainingSet, V, W, Ws, regularisationTerm)
        teError = utils.computeError(testingSet,  V, W, Ws, regularisationTerm, True)
        print("Train error: ", trError, " | Test error: ",  teError)
        
        # Saving the model (every X epoch)
        print("Saving model...")
        vocabulary.vocab.save("save/dict")
        np.savez("save/model", V=V, W=W, Ws=Ws)
        
    print("Training complete, validating...")
    vaError = utils.computeError(validationSet,  V, W, Ws, regularisationTerm, True)
    print("Validation error: ", vaError)
    

if __name__ == "__main__":
    main()
