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
        
        self.level = -1 # Deepness of the node (0=root) (Useful to plot more nicelly)
        
        # Node values
        # Pointer to the vocabulary list
        self.word = None # If the node is a leaf, contains the Word (string and vector representaion) loaded from the dictionary, None otherwise
        self.label = -1 # Sentiment 0-4 (Ground truth)
        
        # For backpropagation:
        self.output  = None # Output of the tensor network AFTER the activation function (same as .word.vect if leaf) (of dimention wordVectSpace)
        
    def labelVect(self):
        """
        Return the ground truth of the label (Variable t on the paper).
        """
        t = np.zeros(params.nbClass) # Zeros everywhere
        t[self.label] = 1 # Except a one on the true label
        return t
    
    def printInd(self, *args):
        """
        Indentate the text to print with the level value (useful to plot the trees)
        """
        for i in range(self.level):
            print('  ', end="")
        for arg in args:
            print(arg, end="")
        print() # New line

class Tree:
    def __init__(self, sentence):
        """
        Generate the tree by parsing the given sentence.
        Args:
            sentence: sentence at the PTB format
        """
        self.root = self._parseSentence(sentence, 0) # Generate the tree
        # self.printTree() # Debug
        
    def _parseSentence(self, sentence, level):
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
        node.level = level
        if positionBreak != 0: # Submatch, continue exploring
            leftSentence = subsequence[0:positionBreak]
            rightSentence = subsequence[positionBreak+1:]
            node.l = self._parseSentence(leftSentence, level+1)
            node.r = self._parseSentence(rightSentence, level+1)
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
    
    def evaluateAllError(self, Ws, verbose=False):
        """
        Return the error recursively (nb of not correctly classified label)
        """
        nbOfNodes = self._nbOfNodes(self.root)
        if verbose:
            print(nbOfNodes, "nodes", "(label,predLab) -> result, cumul")
        return self._evaluateAllError(self.root, Ws, verbose, 0), nbOfNodes
    
    def _evaluateAllError(self, node, Ws, verbose, level):
        # Cost at the current node
        y = utils.softClas(Ws, node.output) # Softmax prediction
        currentCost    = np.log(y[node.label]) # We only take the cell which correspond to the label, all other terms are null
        labelPredicted = np.argmax(y) # Predicted label
        isLabelCorrect = (labelPredicted == node.label)
        nbLabelCorrect = int(isLabelCorrect)

        if verbose: # Plot infos of the current node
            for i in range(level):
                print('  ', end="")
            if node.word is not None: # Not a leaf, we continue exploring the tree
                print(node.word.string, end="")
            print(": C=", currentCost, "(", node.label,  ",", labelPredicted, ") ->", isLabelCorrect)
            print(y)
        if node.word is None: # Not a leaf, we continue exploring the tree
            currentCostL, nbLabelCorrectL = self._evaluateAllError(node.l, Ws, verbose, level+1) # Left
            currentCostR, nbLabelCorrectR = self._evaluateAllError(node.r, Ws, verbose, level+1) # Right
            
            currentCost    += currentCostL    + currentCostR
            nbLabelCorrect += nbLabelCorrectL + nbLabelCorrectR
        return currentCost, nbLabelCorrect
    
    def evaluateRootError(self, Ws):
        """
        Return the error of the complete sentence (at the root) (nb of not correctly classified label)
        """
        pass # TODO Eventually
        
    def printTree(self):
        """
        Recursivelly print the tree
        """
        if(self.root != None):
            print("Tree: ", self.root.label)
            self._printTree(self.root, 0)

    def _printTree(self, node, level):
        if(node != None):
            if node.word is None:
                self._printTree(node.l, level+1)
                self._printTree(node.r, level+1)
            else: # Leaf
                for i in range(level):
                    print('  ', end="")
                print(node.word.string, " ", node.label)

    def _nbOfNodes(self, node):
        if node.word is None:
            return 1 + self._nbOfNodes(node.l) + self._nbOfNodes(node.r)
        return 1
