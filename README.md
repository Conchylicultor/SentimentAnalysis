# SentimentAnalysis
Implementation of the RNTN from Standford (http://nlp.stanford.edu/sentiment/) to detect the sentiments of movie critics.

Requirement:
* Python3
* Numpy
* Matplotlib (for plotting the curves)

The parameters are defined on rntnmodel.py and main.py. To start the training, just type:

`
python3 main.py [optional_save_name]
`

By default, it will save and record the results on '_save/training_'

The project contains different scripts:
* main.py: main loop here: launch the training and record the results for different parameters
* test.py: will use the model present in 'save/' folder to evaluate the cost on the validation set
* unitaryTest: check the gradient by comparing them to the numerical approximation

The structure of the project is as follow:
* train.py: Do one training with the given parameters and record the results
* rntnmodel.py: our model with the gradient computation,...
* tree.py: Contain the tree and node class which correspond to a sentence
* vocabulary.py: Class which contain the dictionary of all the words
* utils.py: some utilities fcts (ex sofmax, loadDataset,...) used by others
