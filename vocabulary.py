#!/usr/bin/env python3

"""
Class which manage all the vector word representations
"""

import numpy as np
import pickle
import params

class Word:
    """
    Turple which contain the word and its space representation
    """
    def __init__(self, word):
        self.string = word
        self.vect = params.randInitMaxValueWords * np.random.rand(params.wordVectSpace) # Initialisation to small values

class Vocab:
    def __init__(self, filename=None):
        self.dictionary = {} # List of Words (dictionary)
        if filename is not None:
            f = open(filename + ".pkl", 'rb')
            self.dictionary = pickle.load(f) # If the file is empty the ValueError will be thrown
            print(len(self.dictionary), " words loaded!")
            
    
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
            found = Word(newWord)
            self.dictionary[newWord] = found
        
        return found
        
    def sort(self):
        """
        Sort the dictionary alphabetically (Usefull ? < Not anymore than we use a dictionary instead of a list)
        """
        print("Nb of words", len(self.dictionary))
        pass
        
    def save(self, filename):
        """
        Sort the dictionary alphabetically (Usefull ? < Not anymore than we use a dictionary instead of a list)
        """
        f = open(filename + ".pkl", 'wb')
        pickle.dump(self.dictionary, f)
                
        
# Global variable which store all the words
vocab = None
def initVocab(): # Is initialized in main.py
    global vocab
    vocab = Vocab()
