#!/usr/bin/env python3

"""
Script to expertiment and verify matrix multiplications as well
as some parts of the script
""" 

import numpy as np
import time # Checking performances
import utils
import params
import tree
import vocabulary
import rntnmodel

def testMath():
    # Test the array multiplication
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    xyT = np.outer(x, y)
    print("x*y': ", xyT)
    print("Type of x*y': ", type(xyT))
    print("Type of x: ", type(x))
    print("Type of y': ", type(y))
    print("Shape of x*y': ", xyT.shape)
    print("Shape of x': ", x.shape)
    print("Shape of y': ", y.shape)
    

# Gradient checking
    
def computeNumericalGradient(sample, model):
    """
    Add and substract epsilon to compute an aproximation of the gradient
    """
    print("Try computing the numerical gradient...")
    
    # Merge all parametters
    initialParams = model.getFlatWeights()
    
    numGrad = np.zeros(initialParams.shape)
    perturb = np.zeros(initialParams.shape)
    epsilon = 1e-6
    
    print(len(numGrad), " params to check (take your time, it will be long...)")
    
    for p in range(len(initialParams)): # Iterate over all our dimentions
        perturb[p] = epsilon # Perturbation on each of the dimentions
        
        # Compute cost at x-e
        model.setFlatWeights(initialParams - perturb)
        loss1 = model.computeError(sample, True)
        
        # Compute cost at x+e
        model.setFlatWeights(initialParams + perturb)
        loss2 = model.computeError(sample, True)
        
        numGrad[p] = (loss2.getRegCost()-loss1.getRegCost())/(2*epsilon) # Derivate approximation
        
        perturb[p] = 0 # Restore to initial value
        
        if p%20 == 0:
            print('Progress:', p,'/', len(initialParams))
    
    model.setFlatWeights(initialParams)
    return model.flatWeigthsToGrad(numGrad) # Return a gradient object
    
def testCheckGradient():
    """
    Gradient checking by comparing to an approximation value
    """
    
    # Create a sample
    sample = tree.Tree("(4 (2 (2 But) (2 (3 (3 (2 believe) (2 it)) (2 or)) (1 not))) (4 (2 ,) (4 (2 it) (4 (4 (2 's) (4 (2 one) (4 (2 of) (4 (4 (2 the) (4 (4 (2 most) (4 (4 beautiful) (3 (2 ,) (3 evocative)))) (2 works))) (2 (2 I) (2 (2 've) (2 seen))))))) (2 .)))))")
    #sample.printTree() # Check parsing and sample loading
    
    # Initialize the model
    params.randInitMaxValueNN = 1.0 # Try bigger values for the initial values
    model = rntnmodel.Model()
    
    # Compute the gradient using the direct formula
    model.evaluateSample(sample)
    analyticGradient = model.backpropagate(sample)
    analyticGradient = model.addRegularisation(analyticGradient, 1) # Don't forget to add the regularisation
    
    # Compute the gradient using the numerical approximation
    numericalGradient = computeNumericalGradient(sample, model)
    
    # Show results (detailled values)
    #print("Computed  V[3]=\n", numericalGradient.dV[3])
    #print("Numerical V[3]=\n", analyticGradient.dV[3])
    #print("Computed  W=\n", numericalGradient.dW)
    #print("Numerical W=\n", analyticGradient.dW)
    #print("Computed  b=\n", numericalGradient.db)
    #print("Numerical b=\n", analyticGradient.db)
    print("Computed  Ws=\n", numericalGradient.dWs)
    print("Numerical Ws=\n", analyticGradient.dWs)
    print("Computed  bs=\n", numericalGradient.dbs)
    print("Numerical bs=\n", analyticGradient.dbs)
    
    # Show results (distance)
    #distV  = np.linalg.norm(analyticGradient.dV  - numericalGradient.dV)  / np.linalg.norm(analyticGradient.dV  + numericalGradient.dV)
    #distW  = np.linalg.norm(analyticGradient.dW  - numericalGradient.dW)  / np.linalg.norm(analyticGradient.dW  + numericalGradient.dW)
    #distb  = np.linalg.norm(analyticGradient.db  - numericalGradient.db)  / np.linalg.norm(analyticGradient.db  + numericalGradient.db)
    distWs = np.linalg.norm(analyticGradient.dWs - numericalGradient.dWs) / np.linalg.norm(analyticGradient.dWs + numericalGradient.dWs)
    distbs = np.linalg.norm(analyticGradient.dbs - numericalGradient.dbs) / np.linalg.norm(analyticGradient.dbs + numericalGradient.dbs)

    #print("Distances: V=", distV)
    #print("Distances: W=", distW)
    #print("Distances: b=", distb)
    print("Distances: Ws=", distWs)
    print("Distances: bs=", distbs)

def testOther():
    """
    Some other tests
    """
    pass
    
def main():
    # Dictionary initialisation
    vocabulary.initVocab()
    
    #testOther()
    testCheckGradient()
    #testMath()
    
    pass


if __name__ == "__main__":
    main()
