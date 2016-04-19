# SentimentAnalysis
Implementation of the RNTN from Standford (http://nlp.stanford.edu/sentiment/) to detect the sentiments of movie critics.

Requirement:
* Python3
* Numpy

The parametters are defined on rntnmodel.py and main.py

The project contains different scripts:
* main.py: main loop here: train the model and record the results
* test.py: will use the model present in 'save/' folder to evaluate the cost on the validation set
* unitaryTest: check the gradient by comparing them to the numerical approximation

The structure of the project is as follow:
* tree.py: Contain the tree and node class which correspond to a sentence
* rntnmodel.py: our model with the gradient computation,...
* vocabulary.py: Class which contain the dictionary of all the words
* utils.py: some utilities fcts (ex sofmax, loadDataset,...) used by others
