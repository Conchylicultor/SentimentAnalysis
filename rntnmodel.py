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
        self.bs = np.random.rand(params.nbClass)                                                       * params.randInitMaxValueNN # Bias of the softmax classifier
        
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
    
    def backpropagate(self, sample):
        """
        Compute the derivate at each level and return the sum of it
        """
        # Notations:
        #   a: Output at root node (after activation)
        #   z: Output before softmax (z=Ws*a)
        #   y: Output after softmax, final prediction (y=softmax(z))
        #   E: Cost of the current prediction (E = cost(softmax(Ws*a)) = cost(y))
        #   t: Gound truth prediction (labelVect)
        # We then have:
        #   t -> a -> t -> a -> ... t -> a(last layer) -> z (projection on dim 5) -> y (softmax prediction) -> E (cost)
        
        return self._backpropagate(sample.root, None) # No incoming error for the root node (except the one coming from softmax)
    
    def _backpropagate(self, node, sigmaCom):
        #node.printInd("Node:")
        #node.printInd("----------")
        
        gradient = ModelGrad() # Store all the gradients
        
        # Compute error coming from the softmax classifier on the current node
        # dE/dz = (t - softmax(z)) Derivative of the cost with respect to the softmax classifier input
        t = node.labelVect()
        y = utils.softClas(self.Ws, node.output)
        dE_dz = (t - y) # TODO: Check the sign !!
        
        #node.printInd("o=", node.output)
        #node.printInd("y=", y)
        #node.printInd("t=", t)
        #node.printInd("dEdz=", dE_dz)
        
        # Gradient of Ws
        gradient.dWs = np.outer(dE_dz, node.output) # (t-y)*a'
        gradient.dbs = dE_dz
        
        #node.printInd("dbs=", gradient.dbs)
        #node.printInd("dWs=", gradient.dWs)
        
        if node.word is None:
            gradient += self._backpropagate(node.l, None)
            gradient += self._backpropagate(node.r, None)
        else: # Leaf: Update L
            pass
            
        
        ## Error coming through the softmax classifier (d*1 vector)
        #sigmaSoft = np.multiply(self.Ws.T.dot(dE_dz), utils.actFctDerFromOutput(node.output)) # Ws' (t_i-y_i) .* f'(x_i) (WARNING: The node.output correspond to the output AFTER the activation fct, so we have f2'(f(x_i)))
        #if sigmaCom is None: # Root node
            #sigmaCom = sigmaSoft # Only softmax error is incoming
        #else:
            #sigmaCom += sigmaSoft # Otherwise, we also add the incoming error from the previous node
        
        #gradientV = None
        #gradientW = None
        
        #if(node.word != None): # Leaf
            ## TODO: Backpropagate L too ??? Modify the vector word space ???
            #node.word.vect -= sigmaCom # Oposite direction of gradient
            #pass # Return empty value (gradient does not depend of V nor W)
        #else: # Go deeper
            ## Construct the incoming output
            #b = node.l.output
            #c = node.r.output
            #bc = np.concatenate((b, c))
            
            ## Compute the gradient at the current node
            #gradientV = np.zeros((params.wordVectSpace, 2*params.wordVectSpace, 2*params.wordVectSpace))
            #for k in range(params.wordVectSpace):
                #gradientV[k] = sigmaCom[k] * np.outer(bc, bc) # 2d*2d matrix
            #gradientW = np.outer(sigmaCom, bc) # d*2d matrix
            
            ## Compute the error at the bottom of the layer
            #sigmaDown = self.W.T.dot(sigmaCom) # (regular term)
            #for k in range(params.wordVectSpace): # Compute S (tensor term)
                #sigmaDown += sigmaCom[k] * (self.V[k] + self.V[k].T).dot(bc)
            #sigmaDown = np.multiply(sigmaDown, utils.actFctDerFromOutput(bc)) # Activation fct
            ## TODO: What about the activation function (here, after, before ?), check content
            
            #d = params.wordVectSpace
            
            ## Propagate the error down to the next nodes
            #gradientVSub, gradientWSub, gradientWsSub = self._backpropagateRntn(node.l, V, W, Ws, sigmaDown[0:d])
            #if gradientVSub is not None:
                #gradientV += gradientVSub
                #gradientW += gradientWSub # If non leaf, gradientWSub shouldn't be null either
            #gradientWs += gradientWsSub
            #gradientVSub, gradientWSub, gradientWsSub = self._backpropagateRntn(node.r, V, W, Ws, sigmaDown[d:2*d])
            #if gradientVSub is not None:
                #gradientV += gradientVSub
                #gradientW += gradientWSub
            #gradientWs += gradientWsSub
        
        return gradient
    
    def addRegularisation(self, gradient, regularisationTerm):
        """
        Add the regularisation term to the givengradient and
        return it (The given gradient is also modified)
        WARNING: Using the formula of the paper, the regularisation
        term is not divided by two so the derivate added here will be
        multiplied x2
        Args:
            regularisationTerm: The lambda term
        """
        factor = 2*regularisationTerm # Factor 2 for the derivate of the square
        
        # Tensor layer
        #gradient.dV  += factor*self.V # Tensor of the RNTN layer
        #gradient.dW  += factor*self.W # Regular term of the RNTN layer
        #gradient.db  += factor*self.b # Bias for the regular term of the RNTN layer
        
        # Softmax
        gradient.dWs += factor*self.Ws # Softmax classifier
        gradient.dbs += factor*self.bs # Bias of the softmax classifier
        
        # Words
        # TODO: What about dL ??
        # gradient.dL
        
        return gradient
    
    def updateWeights(self, gradient):
        """
        Update the weights according to the gradient
        """
        
        # Tensor layer
        #self.V  += gradient.dV
        #self.W  += gradient.dW
        #self.b  += gradient.db
        
        # Softmax
        self.Ws += gradient.dWs
        self.bs += gradient.dbs
        
        # Words
        #for elem in gradient.dL: # Add every word gradient individually
        #    self.L[elem.i,:] += elem.dl


class ModelGrad:
    """
    One struct which contain the differents gradients
    """

    def __init__(self):
        # Tensor layer
        self.dV  = None # Tensor of the RNTN layer
        self.dW  = None # Regular term of the RNTN layer
        self.db  = None # Bias for the regular term of the RNTN layer
        
        # Softmax
        self.dWs = None # Softmax classifier
        self.dbs = None # Bias of the softmax classifier
        
        # Words << Contained in the vocab variable
        self.dL  = None # List of turple (index, dL_i)
        
    def __iadd__(self, gradient):
        """
        Add two gradient together
        """
        # Tensor layer
        #self.dV  += gradient.dV # Tensor of the RNTN layer
        #self.dW  += gradient.dW # Regular term of the RNTN layer
        #self.db  += gradient.db # Bias for the regular term of the RNTN layer
        
        # Softmax (Computed in any case)
        self.dWs += gradient.dWs # Softmax classifier
        self.dbs += gradient.dbs # Bias of the softmax classifier
        
        # Words << Contained in the vocab variable
        if self.dL is None: # Backpropagate the dL gradient on the upper nodes
            self.dL = gradient.dL
        elif gradient.dL is not None: # In this case, we merge the two lists
            self.dL  += gradient.dL
        
        return self
