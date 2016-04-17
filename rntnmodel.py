#!/usr/bin/env python3

"""
Class which define the model and store the parametters
"""

import numpy as np
import params
import utils
import vocabulary

class Model:

    def __init__(self):
        """
        WARNING: Has to be called after loading the vocabulary
        """
        
        # TODO: Possibility of loading from file (default initialize randomly)
        # Initialisation with small values (best solution ??)
        
        # Tensor layer
        self.V  = np.random.rand(params.wordVectSpace, 2*params.wordVectSpace, 2*params.wordVectSpace) * params.randInitMaxValueNN # Tensor of the RNTN layer
        self.W  = np.random.rand(params.wordVectSpace, 2*params.wordVectSpace)                         * params.randInitMaxValueNN # Regular term of the RNTN layer
        self.b  = np.random.rand(params.wordVectSpace)                                                 * params.randInitMaxValueNN # Bias for the regular term of the RNTN layer (WARNING: Confusions with b as input)
        
        # Softmax
        self.Ws = np.random.rand(params.nbClass, params.wordVectSpace)                                 * params.randInitMaxValueNN # Softmax classifier
        self.bs = np.random.rand(params.wordVectSpace)                                                 * params.randInitMaxValueNN # Bias of the softmax classifier
        
        # Words << Contained in the vocab variable 
        self.L  = np.random.normal(0.0, params.randInitMaxValueWords, (vocabulary.vocab.length(), params.wordVectSpace))# Vocabulary (List of N words on vector representation) (Indexing over the first variable: more perfs!)
        

    def evaluateSample(self, sample):
        """
        Evaluate the vector of the complete sentence and compute (and store) all the intermediate
        values (used for backpropagation)
        Compute the output at each node
        """
        self._evaluateNode(sample.root)

    def _evaluateNode(self, node):
        """
        Args:
            Level: Deepeness of the tree (only used for debuging message
        """
        #node.printInd("Node:")
        #node.printInd("----------")
        if node.word is not None: # Leaf
            node.output = self.L[node.word.idx, :]
            #node.printInd(node.word.string)
            #node.printInd(node.output)
            #node.printInd(node.output.shape)
        else: # Go deeper
            # Input
            b = self._evaluateNode(node.l)
            c = self._evaluateNode(node.r)
            
            inputVect = np.concatenate((b, c))
            
            # Compute the tensor term
            tensorResult = np.zeros(params.wordVectSpace)
            for i in range(params.wordVectSpace):
                tensorResult[i] = inputVect.T.dot(self.V[i]).dot(inputVect) # x' * V * x (Compute the tensor layer)
            
            # Compute the regular term
            regularResult = np.dot(self.W,inputVect) + self.b
            
            # Store the result for the backpropagation (AFTER the activation function!!)
            node.output = utils.actFct(tensorResult + regularResult)
            #node.printInd(node.output)
            #node.printInd(node.output.shape)
        return node.output
        
class ModelGrad:
    """
    One struct which contain the differents gradients
    """

    def __init__(self):
        pass
