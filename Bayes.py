
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from First_Dataset import First_Dataset
from sklearn.naive_bayes import GaussianNB
from sklearn.multioutput import MultiOutputClassifier
from Second_Dataset import Second_Dataset
from Data_Preprocessing import *


class BayesClassifier():

	def __init__(self):
		self.model = GaussianNB()

	def fit(self, trainingData, trainingLabels):
		self.model.fit(trainingData.toarray(), trainingLabels)
	
	def test(self, testData):
		return self.model.predict(testData.toarray())        


def calcAccuracy(guesses, correctLabels):
	acc = accuracy_score(correctLabels, guesses) * 100
	print ("Accuracy: ", acc)
	return acc


def runBayes(trainingData, validationData, testingData, 
trainingLabels, validationLabels, testingLabels):


	classifier = BayesClassifier()
	print("Training...")
	classifier.fit(trainingData=trainingData, trainingLabels=trainingLabels)

	print("Validating...")
	guesses = classifier.test(testData=validationData)

	guesses = classifier.test(testingData)
	accuracy = calcAccuracy(guesses=guesses, correctLabels=testingLabels)

	print("Accuracy", accuracy)

				

dataset = First_Dataset()
# dataset = Second_Dataset()
dataset.preprocess()

dataset.split_2()
Training_pad, Validation_pad, Testing_pad, Y_train, Y_val, Y_test = dataset.getData()

X_train_tfidf, X_val_tfidf, X_test_tfidf = convert_to_tfidf(Training_pad, Validation_pad, Testing_pad)


# dataset = First_Dataset()
# dataset.preprocess()
# dataset.encode_labels()
# dataset.embed_tokens(max_words,max_len)
# dataset.split()

# Training_pad, Validation_pad, Testing_pad, Y_train, Y_val, Y_test = dataset.getData()

runBayes (X_train_tfidf, X_val_tfidf, X_test_tfidf, Y_train, Y_val, Y_test)