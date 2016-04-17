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
        # Learning parametters
        
        self.regularisationTerm = 0.0001 # Lambda
        self.learningRate = 0.1 # TODO: Replace by AdaGrad !!
        
        # Weights
        
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
        
    def _predictNode(self, node):
        """
        Return the softmax sentiment prediction for the given word vector
        WARNING: The node output(after activation fct) has to be already computed
        """
        z = np.dot(self.Ws, node.output) + self.bs
        return utils.softmax(z)
        
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
        Compute the derivate at each level of the sample and return the sum
        of it (stored in a gradient object)
        """
        # Notations:
        #   a: Output at root node (after activation)
        #   z: Output before softmax (z=Ws*a + bs)
        #   y: Output after softmax, final prediction (y=softmax(z))
        #   E: Cost of the current prediction (E = cost(softmax(Ws*a + bs)) = cost(y))
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
        y = self._predictNode(node)
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
    
    def addRegularisation(self, gradient, miniBatchSize):
        """
        Add the regularisation term to the givengradient and
        return it (The given gradient is also modified)
        WARNING: Using the formula of the paper, the regularisation
        term is not divided by two so the derivate added here will be
        multiplied x2
        Args:
            regularisationTerm: The lambda term
        """
        factor = 2 * self.regularisationTerm * miniBatchSize # Factor 2 for the derivate of the square
        
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
        #self.V  += self.learningRate * gradient.dV
        #self.W  += self.learningRate * gradient.dW
        #self.b  += self.learningRate * gradient.db
        
        # Softmax
        self.Ws += self.learningRate * gradient.dWs
        self.bs += self.learningRate * gradient.dbs
        
        # Words
        #for elem in gradient.dL: # Add every word gradient individually
        #    self.L[elem.i,:] += self.learningRate * elem.dl
        
    def computeError(self, dataset, compute = False):
        """
        Evaluate the cost error of the given dataset using the parametters
        Args:
            dataset: Collection of the sample to evaluate (can also be a single element)
            compute: If false, the dataset must have completed the forward pass with the given parametters
            before calling this function (the output will not be computed in this fct but the old one will 
            be used)
        Return:
            TODO: Return also the % of correctly classified labels (and the number) (by node ?? or just the root ?? < Both)
            In the paper, they uses 4 metrics (+/- or fine grained ; all or just root)
        """
        
        # If dataset is singleton, we encapsulate is in a list
        if not isinstance(dataset, list):
            dataset = [dataset]
        
        # Evaluate error for each given sample
        error = ModelError() # Will store the different metrics
        for sample in dataset:
            if compute: # If not done yet, compute the Rntn
                self.evaluateSample(sample)
            error += self._evaluateCostNode(sample.root) # Normalize also by number of nodes ?? << Doesn't seems to be the case in the paper
            error.nbOfSample += 1
        
        # Normalize the cost by the number of sample
        error.cost /= error.nbOfSample
        
        # Add regularisation (No regularisation for the bias term)
        costReg = self.regularisationTerm * (np.sum(self.V*self.V) + np.sum(self.W*self.W) + np.sum(self.Ws*self.Ws)) # Numpy array so element-wise multiplication (What about L)
        error.regularisation += costReg # Add regularisation (add N times (for each samples), then normalized)
        
        return error
    
    def _evaluateCostNode(self, node):
        """
        Recursivelly compute the error
        """
        error = ModelError()
        
        # Cost at the current node
        y = self._predictNode(node) # Softmax prediction
        error.cost     = np.log(y[node.label]) # We only take the cell which correspond to the label, all other terms are null
        labelPredicted = np.argmax(y) # Predicted label
        error.nbOfCorrectLabel = int(labelPredicted == node.label)
        error.nbOfNodes = 1

        ## Debug infos
        #if node.word is not None: # Not a leaf, we continue exploring the tree
            #node.printInd(node.word.string)
        #node.printInd("Individual: ", error)
        #node.printInd("(label,prediction) = (", node.label,  ",", labelPredicted, ")")
        #node.printInd(y)
            
        if node.word is None: # Not a leaf, we continue exploring the tree
            error += self._evaluateCostNode(node.l) # Left
            error += self._evaluateCostNode(node.r) # Right
        
        #node.printInd("Collective: ", error)
        return error
    
    def saveModel(self, destination):
        """
        Save the model at the given destination (the destination should not contain
        the extension)
        This save both model parametters and dictionary
        """
        vocabulary.vocab.save(destination + "_dict")
        np.savez(destination + "_model", V=self.V, W=self.W, b=self.b, Ws=self.Ws, bs=self.bs, L=self.L)

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
        
        # Words
        if self.dL is None: # Backpropagate the dL gradient on the upper nodes
            self.dL = gradient.dL
        elif gradient.dL is not None: # In this case, we merge the two lists
            self.dL  += gradient.dL
        
        return self

class ModelError:
    """
    One struct which contain the differents errors (cost, nb of correct predictions,...)
    """

    def __init__(self):
        # Variables allowing us to normalize the cost errors
        self.nbOfNodes  = 0
        self.nbOfSample = 0
        
        self.cost = 0 # Regular cost (formula)
        self.regularisation = 0 # WARNING: Is added only one times for all the samples (avoid adding at each loop)
        self.nbOfCorrectLabel = 0 # Nb of corrected predicted labels
        # Could also add the binary prediction (just +/- and the predictions to the root)
    
    def __str__(self):
        """
        Show diverse informations
        """
        return "Cost=%4f | CostReg=%4f | Percent=%2f%% (%d/%d) | NbOfSamples=%d" % (\
            self.cost/(self.nbOfSample+1), # We add the +1 because if we try to plot inside a tree, it will divide by 0 \
            self.cost + self.regularisation,\
            self.nbOfCorrectLabel*100/self.nbOfNodes,\
            self.nbOfCorrectLabel,\
            self.nbOfNodes,\
            self.nbOfSample)
        
    
    def __iadd__(self, error):
        """
        Add two errors together
        """
        self.nbOfNodes          += error.nbOfNodes
        self.nbOfSample         += error.nbOfSample
        self.cost               += error.cost
        self.regularisation     += error.regularisation
        self.nbOfCorrectLabel   += error.nbOfCorrectLabel
        
        return self
