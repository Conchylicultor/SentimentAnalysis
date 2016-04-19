#!/usr/bin/env python3

"""
Class which manage all the vector word representations
"""

import numpy as np
import pickle

class Word:
    """
    Turple which contain the word and its space representation
    """
    def __init__(self, word, idx):
        self.string = word
        self.idx = idx # Index of the word inside the L matrix

class Vocab:
    def __init__(self, filename=None):
        """
        If the parametters is set, it will try to load the dictionary from
        the given filename
        """
        self.dictionary = {} # List of Words (dictionary)
        self.currentIdx = 0
        if filename is not None:
            f = open(filename + "_dict.pkl", 'rb')
            self.dictionary = pickle.load(f) # If the file is empty the ValueError will be thrown
            f.close()
            print(len(self.dictionary), " words loaded!")
            currentIdx = len(self.dictionary) # In theory, all the words already have been loaded but we never know
            
    
    def addWord(self, newWord):
        """
        Check if the word doesn't exist and eventually add it to the dictionary
        Arg:
            newWord: a string
        Return:
            A reference on the dictionary word
        """
        found = None
        if newWord in self.dictionary: # Does the word is already present ?
            found = self.dictionary[newWord]
        else: # If not found, we create and add it to the dictionary
            # print('Add new word', newWord) # Debug
            found = Word(newWord, self.currentIdx)
            self.dictionary[newWord] = found
            self.currentIdx += 1
        
        return found
        
    def length(self):
        """
        Return the size of the dictionary
        """
        return len(self.dictionary)
        
    def save(self, filename):
        """
        Save the dictionary using pickle
        """
        f = open(filename + "_dict.pkl", 'wb')
        pickle.dump(self.dictionary, f)
        f.close()
                
        
# Global variable which store all the words
vocab = None
def initVocab(filename = None): # Is initialized in main.py
    global vocab
    vocab = Vocab(filename)
