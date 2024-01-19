
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from First_Dataset import First_Dataset
from sklearn.naive_bayes import GaussianNB
from sklearn.multioutput import MultiOutputClassifier


class BayesClassifier():

	def __init__(self):
		self.model = MultiOutputClassifier(GaussianNB(), n_jobs=-1)

	def fit(self, trainingData, trainingLabels):
		self.model.fit(trainingData, trainingLabels)
	
	def test(self, testData):
		return self.model.predict(testData)        


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

				


max_words = 5000
max_len = 50

dataset = First_Dataset()
dataset.preprocess()
dataset.encode_labels()
dataset.embed_tokens(max_words,max_len)
dataset.split()

Training_pad, Validation_pad, Testing_pad, Y_train, Y_val, Y_test = dataset.getData()

runBayes (Training_pad, Validation_pad, Testing_pad, Y_train, Y_val, Y_test)