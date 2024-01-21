
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from First_Dataset import First_Dataset
from sklearn.naive_bayes import GaussianNB
from Second_Dataset import Second_Dataset
from Data_Preprocessing import *
from sklearn.metrics import classification_report

class BayesClassifier():

	def __init__(self):
		self.model = GaussianNB()

	def fit(self, trainingData, trainingLabels):
		self.model.fit(trainingData.toarray(), trainingLabels)
	
	def test(self, testData):
		return self.model.predict(testData.toarray())        



def runBayes(trainingData, validationData, testingData, 
trainingLabels, validationLabels, testingLabels):


	classifier = BayesClassifier()
	print("Training...")
	classifier.fit(trainingData=trainingData, trainingLabels=trainingLabels)

	print("Testing...")

	guesses = classifier.test(testingData)
	print(classification_report(testingLabels, guesses))
				

dataset = First_Dataset()
# dataset = Second_Dataset()
dataset.preprocess()

dataset.split_2()
Training_pad, Validation_pad, Testing_pad, Y_train, Y_val, Y_test = dataset.getData()

X_train_tfidf, X_val_tfidf, X_test_tfidf = convert_to_tfidf(Training_pad, Validation_pad, Testing_pad)

runBayes (X_train_tfidf, X_val_tfidf, X_test_tfidf, Y_train, Y_val, Y_test)