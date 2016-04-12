#!/usr/bin/env python3

"""
Class which contain a sentence on the form of a tree
"""

import params

class Node:
    def __init__(self):
        # Tree structure
        self.p = None # Parent (Usefull ??)
        self.l = None
        self.r = None
        
        # Node values
        self.word = # Word
        # self.v = # Vector representation of the word (of dimention wordVectSpace)
        self.label = # Sentiment 0-4 (Ground truth)
        
        # For backpropagation:
        # self.sigmaCom =  
        # self.sigmaDown = 

class Tree:
    def __init__(self, sentence):
    """
    Generate the tree by parsing the given sentence.
    Args:
        sentence: sentence at the PTB format
    """
        self.root = None
        self.bottom = None # List of nodes of the based words (Useless ??)

    def getRoot(self):
        return self.root

    def deleteTree(self):
        # Garbage collector will do this for us. 
        self.root = None

    def printTree(self):
        if(self.root != None):
            self._printTree(self.root, 0)

    def _printTree(self, node, level):
        """
        Recursivelly print the tree
        """
        if(node != None):
            if(node.l != None):
                self._printTree(node.l, level+1)
            if(node.word != None):
                for i in range(level):
                    print('  ', end="")
                print(node.word)
            if(node.r != None):
                self._printTree(node.l, level+1)
