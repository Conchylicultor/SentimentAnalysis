#!/usr/bin/env python3

"""
Class which contain a sentence on the form of a tree
"""
import re # Regex to parse the file
import numpy as np
import params
import vocabulary
import utils

class Node:
    def __init__(self):
        # Tree structure
        #self.p = None # Parent (Usefull ??)
        self.l = None
        self.r = None
        
        # Node values
        # Pointer to the vocabulary list
        self.word = None # If the node is a leaf, contains the Word (string and vector representaion) loaded from the dictionary, None otherwise
        
        self.label = -1 # Sentiment 0-4 (Ground truth)
        
        # For backpropagation:
        self.output  = None # Output of the tensor network AFTER the activation function (same as .word.vect if leaf) (of dimention wordVectSpace)
        # self.sigmaCom =  
        # self.sigmaDown = 
        
    def labelVect(self):
        """
        Return the ground truth of the label (Variable t on the paper).
        """
        t = np.zeros(params.nbClass) # Zeros everywhere
        t[self.label] = 1 # Except a one on the true label
        return t

class Tree:
    def __init__(self, sentence):
        """
        Generate the tree by parsing the given sentence.
        Args:
            sentence: sentence at the PTB format
        """
        self.root = self._parseSentence(sentence) # Generate the tree
        # self.printTree() # Debug
        
    def _parseSentence(self, sentence):
        """
        Generate the tree from the string
        """
        # Define the patterns
        patternRoot = '\(([0-9]+) (.*)\)' # For the label and the word/subsentence
        
        # Extract infos
        m = re.match(patternRoot, sentence) # Matching (TODO: Compile the regex for better perfs)
        label = int(m.group(1)) # Extract the sentiment label
        subsequence = m.group(2) # Extract the next subsentence (or the final word)
        
        # Try matching deeper nodes
        positionBreak = self._computeSplitPosition(subsequence)# Divide the substring in two
        
        # Node creation
        node = Node()
        node.label = label
        if positionBreak != 0: # Submatch, continue exploring
            leftSentence = subsequence[0:positionBreak]
            rightSentence = subsequence[positionBreak+1:]
            node.l = self._parseSentence(leftSentence)
            node.r = self._parseSentence(rightSentence)
        else: # Otherwise, we have reach an end node (leaf)
            node.word = vocabulary.vocab.addWord(subsequence)
        
        return node

    def _computeSplitPosition(self, subsequence):
        """
        Return the position where the sentence is splited in two (the separation
        beetween the left and right subsentence
        """
        nbBraceOpened = 0
        nbBraceClosed = 0
        positionBreak = 0
        for i in range(len(subsequence)):
            # We count the number of openned and closed brace
            if subsequence[i] == '(':
                nbBraceOpened += 1
            elif subsequence[i] == ')':
                nbBraceClosed += 1
            
            # If those two matchs, then we have found the middle point
            if nbBraceOpened != 0 and nbBraceOpened == nbBraceClosed:
                positionBreak = i+1
                break
        return positionBreak

    def computeRntn(self, V, W):
        """
        Evaluate the vector of the complete sentence and compute all the intermediate
        values (used to backpropagate)
        """
        return self._computeRntn(self.root, V, W)
    
    def _computeRntn(self, node, V, W):
        if(node.word != None): # Leaf
            node.output = node.word.vect
        else: # Go deeper
            # Input
            b = self._computeRntn(node.l, V, W)
            c = self._computeRntn(node.r, V, W)
            
            inputVect = np.concatenate((b, c))
            
            # Compute the tensor term
            tensorResult = np.zeros(params.wordVectSpace)
            for i in range(params.wordVectSpace):
                tensorResult[i] = inputVect.T.dot(V[i]).dot(inputVect) # x' * V * x
            
            node.output = utils.actFct(tensorResult + np.dot(W,inputVect)) # Store the result for the backpropagation (What do we store in the output ?? Before or after the activation fct ??)
        return node.output
    
    def backpropagateRntn(self, Ws):
        """
        Compute the derivate at each level and return the sum of it
        """
        # Notations:
        #   a: Output at root node (after activation)
        #   z: Output before softmax (z=Ws*p)
        #   y: Output after softmax, final prediction (y=softmax(z))
        #   E: Cost of the current prediction (E = cost(softmax(Ws*p)) = cost(y))
        #   t: Gound truth prediction (labelVect)
        # We then have:
        #   t -> a -> t -> a -> ... t -> a(last layer) -> z (projection on dim 5) -> y (softmax prediction) -> E (cost)
        
        return self._backpropagateRntn(self.root, Ws, None) # No incoming error for the root node (except the one coming from softmax)
    
    def _backpropagateRntn(self, node, Ws, sigmaCom):
        # Compute error coming from the softmax classifier on the current node
        # dE/dz = (t - softmax(z)) Derivative of the cost with respect to the softmax classifier input
        t = node.labelVect()
        y = utils.softClas(Ws, node.output)
        dE_dz = (t - y) # TODO: Check the sign !!
        
        # Gradient of Ws
        gradientWs = utils.dotxyt(dE_dz, node.output) # (t-y)*a'
        
        # Error coming through the softmax classifier (d*1 vector)
        sigmaSoft = np.multiply(Ws.T.dot(dE_dz), np.actFctDerFromOutput(node.output)) # Ws' (t_i-y_i) .* f'(x_i) (WARNING: The node.output correspond to the output AFTER the activation fct, so we have f2'(f(x_i)))
        if sigmaCom == None: # Root node
            sigmaCom = sigmaSoft # Only softmax error is incoming
        else:
            sigmaCom += sigmaSoft # Otherwise, we also add the incoming error from the previous node
        
        gradientV = None
        gradientW = None
        
        if(node.word != None): # Leaf
            # TODO: Backpropagate L too ??? Modify the vector word space ???
            pass # Return empty value (gradient does not depend of V nor W)
        else: # Go deeper
            # Construct the incoming output
            b = node.l.output
            c = node.r.output
            bc = np.concatenate((b, c))
            
            # Compute the gradient at the current node
            gradientV = np.zeros((params.wordVectSpace, 2*params.wordVectSpace, 2*params.wordVectSpace))
            for k in range(params.wordVectSpace)
                gradientV[k] = sigmaCom[k] * utils.dotxyt(bc, bc) # 2d*2d matrix
            gradientW = utils.dotxyt(sigmaCom, bc) # d*2d matrix
            
            # Compute the error at the bottom of the layer
            #sigmaDown = 
            #for k in range(params.wordVectSpace)
                #pass
            #sigmaDown = 
            
            # TODO: What about the activation function (here, after, before ?)
            
            # Propagate the error down to the next nodes
            gradientVSub, gradientWSub, gradientWsSub = self._backpropagateRntn(node.l, Ws, None)
            if gradientVSub != None:
                gradientV += gradientVSub
                gradientW += gradientWSub # If non leaf, gradientWSub shouldn't be null either
            gradientWs += gradientWsSub
            gradientVSub, gradientWSub, gradientWsSub = self._backpropagateRntn(node.r, Ws, None)
            if gradientVSub != None:
                gradientV += gradientVSub
                gradientW += gradientWSub
            gradientWs += gradientWsSub
            
            
            #sigmaCom = np.multiply(sigmaCom, utils.actFct) # Activation function
            
            #b = node.l.output
            #c = node.r.output
            #bc = np.concatenate((b, c))
            
            #gradientV = sigmaCom.dot(np.asmatrix(bc).T * np.asmatrix(bc))
            #gradientW = np.asmatrix(sigmaCom).T * np.asmatrix(bc)
            #gradientVLeft,  gradientWLeft  = self._backpropagateRntn(node.l, sigmaDownLeft)
            #gradientVRight, gradientWRight = self._backpropagateRntn(node.r, sigmaDownRight)
            ## TODO: Check if returned gradients are null or not
            #return gradientVLeft + gradientVRight + gradientV, gradientWLeft + gradientWRight + gradientW
        
        return gradientV, gradientW, gradientWs
    
    def evaluateCost(self, Ws):
        """
        Recursivelly compute the cost of each node and sum up the result
        """
        return self._evaluateCost(self.root, Ws)
    
    def _evaluateCost(self, node, Ws):
        #TODO: This could be optimised (only take the right row instead of multiplying the whole matrix)
        currentCost = np.log(utils.softClas(Ws, node.output) [node.label]) # We only take the cell which correspond to the label, all other terms are null
        if(node.word == None): # Not a leaf, we continue exploring the tree
            currentCost += self._evaluateCost(node.l, Ws) # Left
            currentCost += self._evaluateCost(node.r, Ws) # Right
        return currentCost
        
    def printTree(self):
        """
        Recursivelly print the tree
        """
        if(self.root != None):
            print("Tree: ", self.root.label)
            self._printTree(self.root, 0)

    def _printTree(self, node, level):
        if(node != None):
            if(node.l != None):
                self._printTree(node.l, level+1)
            if(node.word != None): # Leaf
                for i in range(level):
                    print('  ', end="")
                print(node.word.string, " ", node.label)
            if(node.r != None):
                self._printTree(node.r, level+1)
