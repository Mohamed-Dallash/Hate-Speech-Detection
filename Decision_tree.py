
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score
from Second_Dataset import Second_Dataset
from First_Dataset import First_Dataset
from Data_Preprocessing import *
from sklearn.metrics import classification_report

tuned_params = [13, 9, 16, 6]

class DecisionTreeClassifier():

	def __init__(self, max_depth = None, min_samples_split = 2, 
	min_samples_leaf = 1, max_leaf_nodes = None):
		self.model = tree.DecisionTreeClassifier(criterion = "entropy", max_depth= max_depth, 
		min_samples_split= min_samples_split, min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes)


	def fit(self, trainingData, trainingLabels):
		# Performing training
		self.model.fit(trainingData, trainingLabels)
	
	def test(self, testData):
		return self.model.predict(testData)
	

	def plotTree(self):
		plt.figure(figsize=(60,20),dpi = 100)
		tree.plot_tree(self.model, filled=True, fontsize=10)
		plt.savefig('decision_tree.png')
		plt.show()        


def calcAccuracy(guesses, correctLabels):
	acc = accuracy_score(correctLabels, guesses) * 100
	print ("Accuracy: ", acc)
	return acc

def classify_with_tuned_params( trainingData, trainingLabels, testingData):
	classifier = DecisionTreeClassifier(tuned_params[0], tuned_params[1], tuned_params[2]
	, tuned_params[3])
	
	classifier.fit(trainingData, trainingLabels)
	classifier.plotTree()

	return classifier.test(testingData)

def runDecisionTree(trainingData, validationData, testingData, 
trainingLabels, validationLabels, testingLabels):

	figure = plt.figure(constrained_layout=True)
	plots = figure.add_gridspec(2, 2)

	hyperparameters = [
		{
			"number" : 0,
			"name": "max depth",			
			"dims":plots[0, 0:1],
			"x_label": "max depth",
			"plot_title":"Accuracy of different max depths",
			"x_content" :list(range(1,15)),
			"values": [[i, 2,1,None] for i in range(1,15)],
			"stats" : [],
		},
		{
			"number" : 1,
			"name": "min samples split",			
			"dims":plots[0, 1:2],
			"x_label": "min samples split",
			"plot_title":"Accuracy of different sample splits",
			"x_content" :list(range(2,15)),
			"values": [[None, i,1,None] for i in range(2,15)],
			"stats" : [],
		},
		{
			"number" : 2,
			"name": "min samples in leaf",			
			"dims":plots[1, 0:1],
			"x_label": "min samples leaf",
			"plot_title":"Accuracy of different samples in leafs",
			"x_content" :list(range(1,15)),
			"values": [[None, 2,i,None] for i in range(1,15)],
			"stats" : [],
		},		
		{
			"number" : 3,
			"name": "max leaf nodes",			
			"dims":plots[1, 1:2],
			"x_label": "max leaf nodes",
			"plot_title":"Accuracy of different leaf nodes",
			"x_content" :list(range(2,15)),
			"values": [[None, 2,1,i] for i in range(2,15)],
			"stats" : [],
		},		

	]

	bestValues = [0,0,0,0]

	for hyperparameter in hyperparameters:
		print("Tuning the ",hyperparameter["name"], " hyperparameter")
		maxAccuracy = 0
		for value in hyperparameter["values"]:
			classifier = DecisionTreeClassifier(value[0], value[1], value[2], value[3])
			print("Training...")
			classifier.fit(trainingData=trainingData, trainingLabels=trainingLabels)

			print("Validating...")
			guesses = classifier.test(testData=validationData)

			accuracy = calcAccuracy(guesses, validationLabels)
			hyperparameter["stats"].append (round(accuracy, 1))
			if (accuracy > maxAccuracy):
				maxAccuracy = accuracy
				bestValues[hyperparameter["number"]] = value[hyperparameter["number"]]

		f = figure.add_subplot(hyperparameter["dims"])

		x = hyperparameter["x_content"]
		y = hyperparameter["stats"]
		f.plot(x,y)
		f.set_xticks(hyperparameter["x_content"])
		f.set_yticks(list(range(0,101,10)))
		for index in range(len(x)):
			f.text(x[index], y[index], y[index], size=10)

		f.set_title(hyperparameter["plot_title"])
		f.set_xlabel(hyperparameter["x_label"])
		f.set_ylabel("Accuracy")
	plt.show()

	print ("---------------------------------------")
	print ("Best values for each hyperparameter: ")
	print("max depth: ", bestValues[0])
	print("minimum number for splitting samples: ", bestValues[1])
	print("minimum samples in a leaf node", bestValues[2])
	print("maximum number of leaf nodes: ", bestValues[3])
	print ("---------------------------------------")

	print("Testing with the tuned hyperparameters...")
	tuned_params= bestValues
	guesses = classify_with_tuned_params( trainingData, trainingLabels, testingData)
	print(classification_report(testingLabels, guesses))



dataset = First_Dataset()
# dataset = Second_Dataset()
dataset.preprocess()

dataset.split_2()
Training_pad, Validation_pad, Testing_pad, Y_train, Y_val, Y_test = dataset.getData()

X_train_tfidf, X_val_tfidf, X_test_tfidf = convert_to_tfidf(Training_pad, Validation_pad, Testing_pad)

runDecisionTree (X_train_tfidf, X_val_tfidf, X_test_tfidf, Y_train, Y_val, Y_test)