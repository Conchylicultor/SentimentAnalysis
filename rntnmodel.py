#!/usr/bin/env python3

"""
Class which define the model and store the parameters
"""

import numpy as np
import utils
import vocabulary
import pickle

class Model:

    def __init__(self, filename=None,
        learningRate=0.1,
        randInitMaxValueNN=0.0001,
        regularisationTerm = 0.0001
            ):
        """
        WARNING: Has to be called after loading the vocabulary
        """
        # Model parameters
        
        self.wordVectSpace = 25 # World vector size
        self.nbClass = 5 # 0-4 sentiments
        
        self.regularisationTerm = regularisationTerm # Lambda
        
        # Learning parameters (AdaGrad)
        
        self.learningRate = learningRate
        self.adagradG = None; # Contain the gradient history
        self.adagradEpsilon = 1e-3;
        
        if filename is None:
            # Initialisation parameters
            
            self.randInitMaxValueWords = 0.1 # For initialize the vector words
            self.randInitMaxValueNN = randInitMaxValueNN # For initialize the NN weights
            
            # Weights
            
            # Initialisation with small values (best solution ??)
            
            # Tensor layer
            self.V  = np.random.rand(self.wordVectSpace, 2*self.wordVectSpace, 2*self.wordVectSpace) * self.randInitMaxValueNN # Tensor of the RNTN layer
            self.W  = np.random.rand(self.wordVectSpace, 2*self.wordVectSpace)                       * self.randInitMaxValueNN # Regular term of the RNTN layer
            self.b  = np.random.rand(self.wordVectSpace)                                             * self.randInitMaxValueNN # Bias for the regular term of the RNTN layer (WARNING: Confusions with b as input)
            
            # Softmax
            self.Ws = np.random.rand(self.nbClass, self.wordVectSpace)                               * self.randInitMaxValueNN # Softmax classifier
            self.bs = np.random.rand(self.nbClass)                                                   * self.randInitMaxValueNN # Bias of the softmax classifier
            
            # Words << Contained in the vocab variable
            self.L  = np.random.normal(0.0, self.randInitMaxValueWords, (vocabulary.vocab.length(), self.wordVectSpace))# Vocabulary (List of N words on vector representation) (Indexing over the first variable: more perfs!)
            
        else: # Loading from file
            print("Loading model from file: ", filename)
            # The dictionary is loaded during initialisation, on the main script
            
            # Loading hyperparameters
            f = open(filename + "_params.pkl", 'rb')
            paramsSave = pickle.load(f)
            f.close()
            
            self.wordVectSpace      = paramsSave["wordVectSpace"] # Useless, could deduce those from the weigths dimentsion
            self.nbClass            = paramsSave["nbClass"] # Useless, same stuff
            self.regularisationTerm = paramsSave["regularisationTerm"]
            
            # Loading weigths
            modelFile = np.load(filename + "_model.npz")
            self.V  = modelFile['V']
            self.W  = modelFile['W']
            self.b  = modelFile['b']
            self.Ws = modelFile['Ws']
            self.bs = modelFile['bs']
            self.L  = modelFile['L']
            
        
    def _predictNode(self, node):
        """
        Return the softmax sentiment prediction for the given word vector
        WARNING: The node output(after activation fct) has to be already
        computed (by the evaluateSample fct)
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
        """
        Same as evaluate sample but for a node (will compute all children recursivelly)
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
            tensorResult = np.zeros(self.wordVectSpace)
            for i in range(self.wordVectSpace):
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
        #   t: Gound truth prediction
        # We then have:
        #   x -> a -> x -> a -> ... x -> a(last layer) -> z (projection on dim 5) -> y (softmax prediction) -> E (cost)
        
        return self._backpropagate(sample.root, None) # No incoming error for the root node (except the one coming from softmax)
    
    def _backpropagate(self, node, sigmaDown):
        #node.printInd("Node:")
        #node.printInd("----------")
        
        gradient = ModelGrad() # Store all the gradients
        
        # Compute error coming from the softmax classifier on the current node
        # dE/dz = (t - softmax(z)) Derivative of the cost with respect to the softmax classifier input
        y = self._predictNode(node)
        y[node.label] -= 1 # = y - t
        dE_dz = y # = (y - t)
        
        #node.printInd("o=", node.output)
        #node.printInd("y=", y)
        #node.printInd("t=", t)
        #node.printInd("dEdz=", dE_dz)
        
        # Gradient of Ws
        gradient.dWs = np.outer(dE_dz, node.output) # (t-y)*aT
        gradient.dbs = dE_dz
        
        #node.printInd("dbs=", gradient.dbs)
        #node.printInd("dWs=", gradient.dWs)
        
        # Error coming through the softmax classifier (d*1 vector)
        sigmaSoft = np.dot(self.Ws.T, dE_dz) # WsT (t_i-y_i)
        
        sigmaCom = sigmaSoft
        if sigmaDown is not None: # Not root node
            sigmaCom += sigmaDown # We also add the incoming error from the upper node
        # Otherwise (root node), only softmax error is incoming
        
        if node.word is None: # Intermediate node, we continue the backpropagation
            # Backpropagate through the activation function
            sigmaCom = np.multiply(sigmaCom, utils.actFctDerFromOutput(node.output)) # sigma .* f'(a_i) (WARNING: The node.output correspond to the output AFTER the activation fct, so we have f2'(f(a_i)))

            # Construct the incoming output
            bc = np.concatenate((node.l.output, node.r.output))
            
            # Compute the gradient of the tensor
            gradient.dV = np.zeros((self.wordVectSpace, 2*self.wordVectSpace, 2*self.wordVectSpace))
            for k in range(self.wordVectSpace):
                gradient.dV[k] = sigmaCom[k] * np.outer(bc, bc) # 2d*2d matrix (*d after the loop)
            gradient.dW = np.outer(sigmaCom, bc) # d*2d matrix
            gradient.db = sigmaCom # d vector
            
            # Compute the error at the bottom of the layer
            sigmaDown = np.dot(self.W.T, sigmaCom) # (regular term)
            for k in range(self.wordVectSpace): # Compute S (tensor term)
                sigmaDown += sigmaCom[k] * (self.V[k] + self.V[k].T).dot(bc)
            
            # Propagate the error down to the next nodes
            d = self.wordVectSpace
            gradient += self._backpropagate(node.l, sigmaDown[0:d])
            gradient += self._backpropagate(node.r, sigmaDown[d:2*d]) # Sum all gradients
        else: # Leaf: Update L
            # If the node is a leaf, we do not go through the activation fct
            
            # dL contain the list of all words which will be modified this pass
            gradient.dL = [ModelDl(node.word.idx, np.copy(sigmaCom))] # Copy probably useless, sigmaCom probably cannot be modified anymore on the other nodes so we could directly pass the reference
        
        return gradient
    
    def addRegularisation(self, gradient, miniBatchSize):
        """
        Add the regularisation term to the given gradient and
        return it (The given gradient is also modified)
        Also normalize the gradient over the miniBatchSize
        WARNING: Using the formula of the paper, the regularisation
        term is not divided by 2 so the derivate added here will be
        multiplied x2 (usefull for gradient checking)
        WARNING: We do not regularize the bias term
        Args:
            gradient: The gradient to regularize
            miniBatchSize: The number of sample taken for this gradient
        """
        factor = 2 * self.regularisationTerm # Factor 2 for the derivate of the square
        
        # Tensor layer
        gradient.dV = gradient.dV/miniBatchSize + factor*self.V
        gradient.dW = gradient.dW/miniBatchSize + factor*self.W
        gradient.db = gradient.db/miniBatchSize
        
        # Softmax
        gradient.dWs = gradient.dWs/miniBatchSize + factor*self.Ws
        gradient.dbs = gradient.dbs/miniBatchSize
        
        # Words
        # TODO: What about dL regularisation ??
        for elem in gradient.dL: # Add every word gradient individually
            elem.g /= miniBatchSize
        
        return gradient
    
    def updateWeights(self, gradient):
        """
        Update the weights according to the gradient
        """
        # Adagrad
        
        # Eventually initialize
        if self.adagradG is None: # First time we update the gradient
            self.adagradG = self.buildEmptyGradient() # The gradient history is stored in a gradient object
            self.adagradG.dL  = np.zeros(self.L.shape) # In the case of the L history, the variable L contain the history of the whole dictionary instead of a list of modification
            
        # We update our gradient history
        self.adagradG.dV  += gradient.dV  * gradient.dV
        self.adagradG.dW  += gradient.dW  * gradient.dW
        self.adagradG.db  += gradient.db  * gradient.db
        self.adagradG.dWs += gradient.dWs * gradient.dWs
        self.adagradG.dbs += gradient.dbs * gradient.dbs
        
        # Step in the opposite direction of the gradient
        self.V  -= self.learningRate * gradient.dV  / np.sqrt(self.adagradG.dV + self.adagradEpsilon)
        self.W  -= self.learningRate * gradient.dW  / np.sqrt(self.adagradG.dW + self.adagradEpsilon)
        self.b  -= self.learningRate * gradient.db  / np.sqrt(self.adagradG.db + self.adagradEpsilon)
        self.Ws -= self.learningRate * gradient.dWs / np.sqrt(self.adagradG.dWs + self.adagradEpsilon)
        self.bs -= self.learningRate * gradient.dbs / np.sqrt(self.adagradG.dbs + self.adagradEpsilon)
        
        # Same thing for the words
        for elem in gradient.dL: # Add every word gradient individually
            self.adagradG.dL[elem.idx,:] += elem.g * elem.g # We add the current word gradient to the adagrad history
            self.L[elem.idx,:] -= self.learningRate * elem.g / np.sqrt(self.adagradG.dL[elem.idx,:] + self.adagradEpsilon) # Update with AdaGrad
    
    def buildEmptyGradient(self):
        """
        Just construct and return an empty gradient
        Note: This function could be replaced by implementing a smarter ModelGrad.__iadd__ which should concider the None case
        """
        gradient = ModelGrad()
        gradient.dV  = np.zeros(self.V.shape)
        gradient.dW  = np.zeros(self.W.shape)
        gradient.db  = np.zeros(self.b.shape)
        gradient.dWs = np.zeros(self.Ws.shape)
        gradient.dbs = np.zeros(self.bs.shape)
        gradient.dL  = []
        
        return gradient
        
        
    def resetAdagrad(self):
        """
        Just restore the AdaGrad history to its initial state
        """
        print("Reset AdaGrad")
        self.adagradG = None # Erase history
        
    def computeError(self, dataset, compute = False):
        """
        Evaluate the cost error of the given dataset using the parameters
        Args:
            dataset: Collection of the sample to evaluate (can also be a single element)
            compute: If false, the dataset must have completed the forward pass with the given parameters
            before calling this function (the output will not be computed in this fct but the old one will 
            be used)
        Return:
            Return an error object which contain the % of correctly classified labels (and the number) (both by nodes and just root)
            In the paper, they uses 4 metrics (+/- or fine grained ; all or just root)
        """
        
        # If dataset is a singleton, we encapsulate it in a list
        if not isinstance(dataset, list):
            dataset = [dataset]
        
        # Evaluate error for each given sample
        error = ModelError() # Will store the different metrics
        for sample in dataset:
            if compute: # If not done yet, compute the Rntn outputs
                self.evaluateSample(sample)
            error += self._evaluateCostNode(sample.root, True) # Normalize also by number of nodes ?? << Doesn't seems to be the case in the paper
            error.nbOfSample += 1
        
        # Add regularisation (No regularisation for the bias terms)
        costReg = self.regularisationTerm * (np.sum(self.V*self.V) + np.sum(self.W*self.W) + np.sum(self.Ws*self.Ws)) # Numpy array so element-wise multiplication (What about L ??)
        error.regularisation += costReg * error.nbOfSample # Add regularisation (add N times (for each samples))
        
        return error
    
    def _evaluateCostNode(self, node, isRoot=False):
        """
        Recursivelly compute the error(s)
        """
        error = ModelError()
        
        # Cost at the current node
        y = self._predictNode(node) # Softmax prediction
        error.cost = -np.log(y[node.label]) # We only take the cell which correspond to the label, all other terms are null
        labelPredicted = np.argmax(y) # Predicted label (0-4)
        sucess = int(labelPredicted == node.label)
        error.nbNodeCorrect = sucess
        if isRoot:
            error.nbRootCorrect = sucess # Add the sucess at the top node
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
    
    # Three functions useful for Gradient Checking
    
    def getFlatWeights(self):
        """
        Return all params concatenated in a big 1d array
        """
        weights = np.concatenate((
            self.V.ravel(),
            self.W.ravel(),
            self.b.ravel(),
            self.Ws.ravel(),
            self.bs.ravel()
            ))
        # TODO: Try on L
        return weights
    
    def setFlatWeights(self, weights):
        """
        Restore the given weights from the given big 1d array
        """
        endIdx = 0 # Useful when commenting (for partial gradient checking)
        
        initIdx = 0
        endIdx = self.V.size
        self.V = np.reshape(weights[initIdx:endIdx], self.V.shape)
        
        initIdx += self.V.size
        endIdx  += self.W.size
        self.W = np.reshape(weights[initIdx:endIdx], self.W.shape)
        
        initIdx += self.W.size
        endIdx  += self.b.size
        self.b = np.reshape(weights[initIdx:endIdx], self.b.shape)
        
        initIdx += self.b.size
        endIdx  += self.Ws.size
        self.Ws = np.reshape(weights[initIdx:endIdx], self.Ws.shape)
        
        initIdx += self.Ws.size
        endIdx  += self.bs.size
        self.bs = np.reshape(weights[initIdx:endIdx], self.bs.shape)
        
    def flatWeigthsToGrad(self, flatWeigths):
        """
        Convert the given weights to a gradient object
        """
        gradient = ModelGrad()
        endIdx = 0 # Useful when commenting (for partial gradient checking)
        
        initIdx = 0
        endIdx = self.V.size
        gradient.dV = np.reshape(flatWeigths[initIdx:endIdx], self.V.shape)
        
        initIdx += self.V.size
        endIdx  += self.W.size
        gradient.dW = np.reshape(flatWeigths[initIdx:endIdx], self.W.shape)
        
        initIdx += self.W.size
        endIdx  += self.b.size
        gradient.db = np.reshape(flatWeigths[initIdx:endIdx], self.b.shape)
        
        initIdx += self.b.size
        endIdx  += self.Ws.size
        gradient.dWs = np.reshape(flatWeigths[initIdx:endIdx], self.Ws.shape)
        
        initIdx += self.Ws.size
        endIdx  += self.bs.size
        gradient.dbs = np.reshape(flatWeigths[initIdx:endIdx], self.bs.shape)
        
        return gradient
    
    def gradToFlatWeigths(self, gradient):
        """
        Return all params concatenated in a big 1d array (gradient version)
        """
        weights = np.concatenate((
            gradient.dV.ravel(),
            gradient.dW.ravel(),
            gradient.db.ravel(),
            gradient.dWs.ravel(),
            gradient.dbs.ravel()
            ))
        # TODO: Try on L
        return weights
    
    # Other utils fcts
    
    def saveModel(self, destination):
        """
        Save the model at the given destination (the destination should not contain
        the extension)
        This save both model parameters and dictionary
        """
        # Vocabulary
        vocabulary.vocab.save(destination) # The fct will add the extention
        
        # Hyperparameters
        paramsSave = {
            'wordVectSpace':      self.wordVectSpace,
            'nbClass':            self.nbClass,
            'regularisationTerm': self.regularisationTerm,
            }
        f = open(destination + "_params.pkl", 'wb')
        pickle.dump(paramsSave, f)
        f.close()
        
        # Weights
        np.savez(destination + "_model",
            V=self.V,
            W=self.W,
            b=self.b,
            Ws=self.Ws,
            bs=self.bs,
            L=self.L)

class ModelDl:
    """
    Struct which represent a word gradient
    """
    def __init__(self, idx, g):
        self.idx = idx # The word id to modify
        self.g = g # Its gradient (vector dim d)
    
class ModelGrad:
    """
    Struct which contain the differents gradients
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
        self.dL  = [] # List of ModelDl (index, dL_i)
        
    def __iadd__(self, gradient):
        """
        Add two gradient together
        """
        # Tensor layer
        if gradient.dV is not None: # Case for the leaf (indead, only depend of the softmax error so no tensor gradient is set)
            self.dV  += gradient.dV # Tensor of the RNTN layer
            self.dW  += gradient.dW # Regular term of the RNTN layer
            self.db  += gradient.db # Bias for the regular term of the RNTN layer
        
        # Softmax (Computed in any case)
        self.dWs += gradient.dWs # Softmax classifier
        self.dbs += gradient.dbs # Bias of the softmax classifier
        
        # Words
        self.dL  += gradient.dL # We merge the two lists (Backpropagate the dL gradient on the upper nodes)
        
        return self

class ModelError:
    """
    Struct which contain the differents errors (cost, nb of correct predictions,...)
    """

    def __init__(self):
        # Variables allowing us to normalize the cost errors
        self.nbOfNodes  = 0
        self.nbOfSample = 0
        
        self.cost = 0 # Regular cost (formula)
        self.regularisation = 0 # WARNING: Is added only one times for all the samples (avoid adding at each loop)
        self.nbNodeCorrect = 0 # Nb of corrected predicted labels
        self.nbRootCorrect = 0 # Nb of corrected predicted labels (only the top level)
        # Could also add the binary prediction (just +/- and the predictions to the root)
    
    def __str__(self):
        """
        Show diverse informations (is called when trying to print the error)
        In the current version, it is not possible to plot inside a tree for debugging (crash when
        divide by 0), it is quite easy to correct though if really needed
        """
        return "Cost=%4f | CostReg=%4f | Percent=%2f%% (%d/%d) | Percent(Root)=%2f%% (%d/%d)" % (
            self.cost/self.nbOfSample,
            self.getRegCost(),
            self.getPercentNodes(),
            self.nbNodeCorrect,
            self.nbOfNodes,
            self.getPercentRoot(),
            self.nbRootCorrect,
            self.nbOfSample)
    
    def toCsv(self):
        """
        Return a string to be saved into a .csv file
        Format: costReg|percentCorrectNodes|percentCorrectRoot
        """
        return "%4f|%4f|%4f" % (self.getRegCost(), self.getPercentNodes(), self.getPercentRoot())
    
    def getRegCost(self):
        """
        Just return the cost with the regularisation term (Normalised by the number of sample)
        """
        assert self.nbOfSample > 0 # Could made the program crash if we try to plot the error while computing it (when debugging the node error)
        return (self.cost + self.regularisation)/self.nbOfSample # If we try to plot inside a tree, it will divide by 0
    
    def getPercentNodes(self):
        """
        Percentage of correctly labelled nodes (all tree nodes taken)
        """
        return self.nbNodeCorrect*100/self.nbOfNodes
    
    def getPercentRoot(self):
        """
        Percentage of correctly labelled samples (only root taken)
        """
        assert self.nbOfSample > 0 # Could made the program crash if we try to plot the error while computing it (when debugging the node error)
        return self.nbRootCorrect*100/self.nbOfSample
    
    def __iadd__(self, error):
        """
        Add two errors together
        """
        self.nbOfNodes          += error.nbOfNodes
        self.nbOfSample         += error.nbOfSample
        self.cost               += error.cost
        self.regularisation     += error.regularisation
        self.nbNodeCorrect      += error.nbNodeCorrect
        self.nbRootCorrect      += error.nbRootCorrect
        
        return self
