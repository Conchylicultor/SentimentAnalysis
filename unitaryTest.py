#!/usr/bin/env python3

"""
Script to expertiment and verify the gradients as well
as some parts of the script
""" 

import numpy as np
import time # Checking performances
import utils
import tree
import vocabulary
import rntnmodel
    

# Gradient checking
    
def computeNumericalGradient(sample, model):
    """
    Add and substract epsilon to compute an aproximation of the gradient
    """
    print("Try computing the numerical gradient...")
    
    # Merge all parametters
    initialParams = model.getFlatWeights()
    
    numGrad = np.zeros(initialParams.shape)
    epsilon = 1e-6
    
    print(len(numGrad), " params to check (take your time, it will be long...)")
    
    for p in range(len(initialParams)): # Iterate over all our dimentions
        save = initialParams[p]
        
        # Compute cost at x-e
        initialParams[p] -= epsilon# Perturbation on each of the dimentions
        model.setFlatWeights(initialParams)
        loss1 = model.computeError(sample, True)
        
        # Compute cost at x+e
        initialParams[p] = save + epsilon
        model.setFlatWeights(initialParams)
        loss2 = model.computeError(sample, True)
        
        initialParams[p] = save # Restore to initial value
        
        numGrad[p] = (loss2.getRegCost()-loss1.getRegCost())/(2*epsilon) # Derivate approximation
        
        if p%300 == 0:
            print('Progress:', p,'/', len(initialParams))
    
    model.setFlatWeights(initialParams)
    return model.flatWeigthsToGrad(numGrad) # Return a gradient object
    
def testCheckGradient():
    """
    Gradient checking by comparing to an approximation value
    """
    
    # Create an arbitrary sample
    sample = tree.Tree("(4 (2 (2 But) (2 (3 (3 (2 believe) (2 it)) (2 or)) (1 not))) (4 (2 ,) (4 (2 it) (4 (4 (2 's) (4 (2 one) (4 (2 of) (4 (4 (2 the) (4 (4 (2 most) (4 (4 beautiful) (3 (2 ,) (3 evocative)))) (2 works))) (2 (2 I) (2 (2 've) (2 seen))))))) (2 .)))))")
    #sample.printTree() # Check parsing and sample loading
    
    # Initialize the model
    model = rntnmodel.Model(\
        randInitMaxValueNN = 2.0,  # Try bigger values for the initial values \
        #regularisationTerm = 0 # Check without regularisation \
        regularisationTerm = 0.01 # Check gradient with regularisation \
        )
    
    # Compute the gradient using the direct formula
    model.evaluateSample(sample)
    analyticGradient = model.backpropagate(sample)
    analyticGradient = model.addRegularisation(analyticGradient, 1) # Don't forget to add the regularisation
    
    # Compute the gradient using the numerical approximation
    numericalGradient = computeNumericalGradient(sample, model)
    
    # Show results (detailled values)
    print("Computed  dV[3]=\n", numericalGradient.dV[3])
    print("Numerical dV[3]=\n", analyticGradient.dV[3]) # We plot a random layer instead of the whole tensor
    print("Computed  dW=\n", numericalGradient.dW)
    print("Numerical dW=\n", analyticGradient.dW)
    print("Computed  db=\n", numericalGradient.db)
    print("Numerical db=\n", analyticGradient.db)
    print("Computed  dWs=\n", numericalGradient.dWs)
    print("Numerical dWs=\n", analyticGradient.dWs)
    print("Computed  dbs=\n", numericalGradient.dbs)
    print("Numerical dbs=\n", analyticGradient.dbs)
    
    # Show results (distance)
    distV  = np.linalg.norm(analyticGradient.dV  - numericalGradient.dV)  / np.linalg.norm(analyticGradient.dV  + numericalGradient.dV)
    distW  = np.linalg.norm(analyticGradient.dW  - numericalGradient.dW)  / np.linalg.norm(analyticGradient.dW  + numericalGradient.dW)
    distb  = np.linalg.norm(analyticGradient.db  - numericalGradient.db)  / np.linalg.norm(analyticGradient.db  + numericalGradient.db)
    distWs = np.linalg.norm(analyticGradient.dWs - numericalGradient.dWs) / np.linalg.norm(analyticGradient.dWs + numericalGradient.dWs)
    distbs = np.linalg.norm(analyticGradient.dbs - numericalGradient.dbs) / np.linalg.norm(analyticGradient.dbs + numericalGradient.dbs)

    print("Distances: dV=", distV)
    print("Distances: dW=", distW)
    print("Distances: db=", distb)
    print("Distances: dWs=", distWs)
    print("Distances: dbs=", distbs)

    
def main():
    # Dictionary initialisation
    vocabulary.initVocab()
    testCheckGradient()

if __name__ == "__main__":
    main()
