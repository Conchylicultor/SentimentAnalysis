#!/usr/bin/env python3

"""
Main script

Use python 3
"""

import os, sys
import numpy as np
import params
import utils

# Parametters
nbEpoch = 150
learningRate = 0.1 # TODO: Replace by AdaGrad !!

def main():
    # Loading dataset
    
    # Datatransform (normalisation, remove outliers,...)
    
    # Creating the model
    # TODO: Eventually load from file (default initialize randomly)
    #V # Tensor of the RNTN layer
    #W = np. # Regular term of the RNTN layer
    Ws = np.random.rand(nbClass, wordVectSpace) # Softmax classifier # TODO: Initialize randomly ??
    #L = # Vocabulary
    
    # TODO: Include the training in the cross-validation loop
    # Main loop
    for i in range(nbEpoch):
        print("Epoch: ", i)
        
        # Randomly shuffle the dataset
        
        # TODO: Loop over the training samples
        # Select the training sample
        trainingSample = dataset.nextSample()
        
        # Forward pass
        rntnOutput = trainingSample.tree.computeRntn() # Evaluate the model recursivelly
        finalOutput = softmax(Ws * rntnOutput) # Use softmax classifier to get the final prediction
        
        # Backward pass (Compute the gradients)
        gradientWs = np.multiply(trainingSample.labelVect, (np.ones((nbClass, 1)) - softmax(Ws*rntnOutput))) * np.transpose(rntnOutput)
        
        # Update the weights
        Ws -= learningRate * gradientWs # Step in the oposite of the gradient
        
        # Compute new testing error
        
        # Saving the model (every X epoch)


if __name__ == "__main__":
    main()
