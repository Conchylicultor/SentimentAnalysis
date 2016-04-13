#!/usr/bin/env python3

"""
Class which manage all the vector word representations
"""

import numpy as np
import params

class Word:
    """
    Turple which contain the word and its space representation
    """
    def __init__(self, word):
        self.string = word
        self.vect = params.randInitialMaxValue * np.random.rand(params.wordVectSpace) # Initialisation to small values

class Vocab:
    def __init__(self):
        self.dictionary = [] # List of Words
    
    def addWord(self, newWord):
        """
        Check if the word doesn't exist and eventually add it to the dictionary
        Arg:
            newWord: a string
        Return:
            A reference on the dictionary word
        """
        found = None
        for word in self.dictionary: # Does the word is already present ?
            if word.string == newWord:
                found = word
                break # No need to go futher
        
        if found == None: # If not found, we create and add it to the dictionary
            print('Add new word', newWord) # Debug
            found = Word(newWord)
            self.dictionary.append(found)
        
        return found
        
    def sort(self):
        """
        Sort the dictionary alphabetically (Usefull ?)
        """
        print("Nb of words", len(self.dictionary))
        pass
                
        

vocab = None
def initVocab(): # Is initialized in main.py
    global vocab
    vocab = Vocab()
