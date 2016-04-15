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
        self.output  = None # Output of the tensor network after the activation function (same as .word.vect if leaf) (of dimention wordVectSpace)
        # self.sigmaCom =  
        # self.sigmaDown = 

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
        #   p: Output at root node (rntnOutput)
        #   q: Output before softmax (q=Ws*p)
        #   E: Cost of the current prediction (E = cost(softmax(Ws*p)))
        #   t: Gound truth prediction (labelVect)
        # We then have:
        #   p -> ... -> p(last layer) -> q -> E
        
        # dE/dq = t.*(1 - softmax(q)) Derivative of the softmax classifier error
        dE_dq = np.multiply(trainingSample.labelVect(), (np.ones(params.nbClass) - utils.softClas(Ws, rntnOutput)))
        
        # dE/dWs = dE/dq * dq/dWs (with dq/dWs = p')
        gradientWs = np.asmatrix(dE_dq).T * np.asmatrix(rntnOutput) # WARNING: Numpy array does not conserve the orientations so we need to convert to matrices
        
        # dE/dW, dE/dV = dE/dq * dq/dp * dp/dV (same for W)
        dE_dp = Ws.T * dE_dq # WARNING: DOES NOT CORRESPOND EXACTLY TO THE FORMULA ON THE PAPER
        
        sigmaCom = dE_dp
        return self._backpropagateRntn(self.root, sigmaCom)
    
    def _backpropagateRntn(self, node, sigmaCom):
        if(node.word != None): # Leaf
            # TODO: Backpropagate L too ??? Modify the vector word space ???
            return None, None # Return empty value (does not depend of V nor W)
        else: # Go deeper
            sigmaCom = np.multiply(sigmaCom, utils.actFct) # Activation function
            
            b = node.l.output
            c = node.r.output
            bc = np.concatenate((b, c))
            
            gradientV = sigmaCom.dot(np.asmatrix(bc).T * np.asmatrix(bc))
            gradientW = np.asmatrix(sigmaCom).T * np.asmatrix(bc)
            gradientVLeft,  gradientWLeft  = self._backpropagateRntn(node.l, sigmaDownLeft)
            gradientVRight, gradientWRight = self._backpropagateRntn(node.r, sigmaDownRight)
            return gradientVLeft + gradientVRight + gradientV, gradientWLeft + gradientWRight + gradientW
    
    def evaluateCost(self, Ws):
        """
        Recursivelly compute the cost of each node and sum up the result
        """
        return self._evaluateCost(self.root, Ws)
    
    def _evaluateCost(self, node, Ws):
        currentCost = np.log(utils.softClas(Ws, node.output) [node.label]) # We only take the cell which correspond to the label, all other terms are null
        if(node.word == None): # Not a leaf, we continue exploring the tree
            currentCost += self._evaluateCost(node.l, Ws) # Left
            currentCost += self._evaluateCost(node.r, Ws) # Right
        return currentCost
    
    def labelVect(self):
        """
        Return the ground truth of the label (Variable t on the paper).
        """
        t = np.zeros(params.nbClass) # Zeros everywhere
        t[self.root.label] = 1 # Except a one on the true label
        return t
        
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
