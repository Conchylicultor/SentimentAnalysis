#!/usr/bin/env python3

"""
Main script

Use python 3
"""

import os, sys
import numpy as np
import params
import utils
import vocabulary

# Parametters
nbEpoch = 150
learningRate = 0.1 # TODO: Replace by AdaGrad !!
regularisationTerm = 0.01 # Lambda

def main():
    print("Welcome into RNTN implementation 0.1")
    
    print("Parsing dataset, creating dictionary...")
    # Dictionary initialisation
    vocabulary.initVocab()
    
    # Loading dataset
    trainingSet = utils.loadDataset("trees/train.txt");
    testingSet = utils.loadDataset("trees/test.txt");
    #validationSet = loadDataset("trees/dev.txt");
    
    vocabulary.vocab.sort();
    
    print("Datasets loaded !")
    
    # Datatransform (normalisation, remove outliers,...)
    
    # Creating the model
    # TODO: Eventually load from file (default initialize randomly)
    #V # Tensor of the RNTN layer
    #W = np. # Regular term of the RNTN layer
    Ws = np.random.rand(params.nbClass, params.wordVectSpace) # Softmax classifier # TODO: Initialize randomly ??
    #L = # Vocabulary (List of N words on vector, representation)
    
    # TODO: Include the training in the cross-validation loop
    # Main loop
    for i in range(nbEpoch):
        print("Epoch: ", i)
        
        # Randomly shuffle the dataset
        
        # TODO: Loop over the training samples
        # Select the training sample
        trainingSample = trainingSet.nextSample()
        
        # Forward pass
        rntnOutput = trainingSample.tree.computeRntn() # Evaluate the model recursivelly
        finalOutput = utils.softmax(Ws * rntnOutput) # Use softmax classifier to get the final prediction
        
        # Backward pass (Compute the gradients)
        gradientWs = np.multiply(trainingSample.labelVect, (np.ones((nbClass, 1)) - utils.softmax(Ws*rntnOutput))) * np.transpose(rntnOutput)
        gradientWs += regularisationTerm * Ws
        
        # Update the weights
        Ws -= learningRate * gradientWs # Step in the oposite of the gradient
        
        # Compute new testing error
        
        # Saving the model (every X epoch)


if __name__ == "__main__":
    main()
