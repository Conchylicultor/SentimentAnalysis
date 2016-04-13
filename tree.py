#!/usr/bin/env python3

"""
Class which contain a sentence on the form of a tree
"""
import re # Regex to parse the file
import params
import vocabulary

class Node:
    def __init__(self):
        # Tree structure
        #self.p = None # Parent (Usefull ??)
        self.l = None
        self.r = None
        
        # Node values
        # Pointer to the vocabulary list
        self.word = None # If the node is a leaf, contains the Word loaded from the dictionary, None otherwise
        # self.vect = # Vector representation of the word (of dimention wordVectSpace) << Warning: already contained on the word variable !!!!!)
        
        self.label = -1 # Sentiment 0-4 (Ground truth)
        
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
        self.root = self._parseSentence(sentence) # Generate the tree
        self.printTree() # Debug
        
    def _parseSentence(self, sentence):
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

    def computeRntn(self):
        """
        Evaluate the vector of the complete sentence
        """
        return self._computeRntn(self.root)
    
    def _computeRntn(self, node):
        if(node.word != None): # Leaf
            return node.wordVect
        else: # Go deeper
            return model.rntn(self._computeRntn(self.node.l), self._computeRntn(self.node.r))
    
    def printTree(self):
        if(self.root != None):
            print("Tree: ", self.root.label)
            self._printTree(self.root, 0)

    def _printTree(self, node, level):
        """
        Recursivelly print the tree
        """
        if(node != None):
            if(node.l != None):
                self._printTree(node.l, level+1)
            if(node.word != None): # Leaf
                for i in range(level):
                    print('  ', end="")
                print(node.word.string, " ", node.label)
            if(node.r != None):
                self._printTree(node.r, level+1)
