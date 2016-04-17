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

def restoreParams(parametters, shapeV, shapeW, shapeWs):
    sizeV = np.prod(shapeV)
    sizeW = np.prod(shapeW)
    sizeWs = np.prod(shapeWs)
    V  = np.reshape(parametters[          0:sizeV             ], shapeV)
    W  = np.reshape(parametters[      sizeV:sizeV+sizeW       ], shapeW)
    Ws = np.reshape(parametters[sizeV+sizeW:sizeV+sizeW+sizeWs], shapeWs)
    
    #print("Shape of V: ", V.shape)
    #print("Shape of W': ", W.shape)
    #print("Shape of Ws': ", Ws.shape)
    
    return V, W, Ws
    
def computeNumericalGradient(sample, initialV, initialW, initialWs):
    """
    Add and substract epsilon to compute an aproximation of the gradient
    """
    
    # Save shape for later restoration
    shapeV = initialV.shape
    shapeW = initialW.shape
    shapeWs = initialWs.shape
    
    # Merge all parametters
    initialParams = np.concatenate((initialV.ravel(), initialW.ravel(), initialWs.ravel()))
    
    numGrad = np.zeros(initialParams.shape)
    perturb = np.zeros(initialParams.shape)    
    epsilon = 1e-4
    
    for p in range(len(initialParams)): # Iterate over all our dimentions
        perturb[p] = epsilon # Perturbation on each of the dimentions
        
        # Compute cost at x-e
        V, W, Ws = restoreParams(initialParams - perturb, shapeV, shapeW, shapeWs)
        sample.computeRntn(V, W)
        loss1 = sample.evaluateCost(Ws)
        
        # Compute cost at x+e
        V, W, Ws = restoreParams(initialParams + perturb, shapeV, shapeW, shapeWs)
        sample.computeRntn(V, W)
        loss2 = sample.evaluateCost(Ws)
        
        numGrad[p] = (loss2-loss1)/(2*epsilon) # Derivate approximation
        
        perturb[p] = 0 # Restore to initial value
        
        if p%100 == 0:
            print(p)
    
    numGradV, numGradW, numGradWs = restoreParams(numGrad, shapeV, shapeW, shapeWs) # Extract all gradients
    return numGradV, numGradW, numGradWs
    
def testGradient():
    """
    Gradient checking by comparing to an approximation value
    """
    
    # Create a sample
    sample = tree.Tree("(4 (2 (2 But) (2 (3 (3 (2 believe) (2 it)) (2 or)) (1 not))) (4 (2 ,) (4 (2 it) (4 (4 (2 's) (4 (2 one) (4 (2 of) (4 (4 (2 the) (4 (4 (2 most) (4 (4 beautiful) (3 (2 ,) (3 evocative)))) (2 works))) (2 (2 I) (2 (2 've) (2 seen))))))) (2 .)))))")
    #sample.printTree() # Check parsing and sample loading
    
    # Random values between [0-1]
    V  = np.random.rand(params.wordVectSpace, 2*params.wordVectSpace, 2*params.wordVectSpace) # Tensor of the RNTN layer
    W  = np.random.rand(params.wordVectSpace, 2*params.wordVectSpace) # Regular term of the RNTN layer
    Ws = np.random.rand(params.nbClass, params.wordVectSpace) # Softmax classifier

    # Compute the gradient using the direct formula
    sample.computeRntn(V, W)
    sample.evaluateAllError(Ws, verbose=True) # Check the error at each node
    compGradV, compGradW, compGradWs = sample.backpropagateRntn(V, W, Ws)
    
    # Compute the gradient using the numerical approximation
    numGradV, numGradW, numGradWs = computeNumericalGradient(sample, V, W, Ws)
    
    # Show results
    print("Computed  V[3]=\n", compGradV[3])
    print("Numerical V[3]=\n", numGradV[3])
    print("Computed  W=\n", compGradW)
    print("Numerical W=\n", numGradW)
    print("Computed  Ws=\n", compGradWs)
    print("Numerical Ws=\n", numGradWs)
    
    # Use a some metric to compare
    distV  = np.linalg.norm(compGradV  - numGradV)  / np.linalg.norm(compGradV  + numGradV)
    distW  = np.linalg.norm(compGradW  - numGradW)  / np.linalg.norm(compGradW  + numGradW)
    distWs = np.linalg.norm(compGradWs - numGradWs) / np.linalg.norm(compGradWs + numGradWs)
    print("Distances: V=", distV, "W=", distW, "Ws=", distWs)

def testOther():
    """
    Some other tests
    """
    pass
    
def main():
    # Dictionary initialisation
    vocabulary.initVocab()
    
    #testOther()
    testGradient()
    #testMath()
    
    pass


if __name__ == "__main__":
    main()
