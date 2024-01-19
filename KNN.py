from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from matplotlib import pyplot as plt
from First_Dataset import First_Dataset


class KNN():
	
	def __init__(self, n_neighbors, distance_metric) -> None:
		self.model = KNeighborsClassifier(n_neighbors = n_neighbors, metric = distance_metric)
		
	def fit(self, training_data, training_labels):
		self.model.fit(training_data, training_labels)

	
	def test(self, testing_data):
		return self.model.predict(testing_data)
	

def classify_with_tuned_params(trainingData, trainingLabels, testingData, tuned_params):
	classifier = KNN(tuned_params[0], "euclidean")
	classifier.fit(trainingData, trainingLabels)
	return classifier.test(testingData)

def calcAccuracy(guesses, correctLabels):
	acc = accuracy_score(correctLabels, guesses) * 100
	print ("Accuracy: ", acc)
	return acc

def runKNN(trainingData, validationData, testingData, 
trainingLabels, validationLabels, testingLabels):
	
	figure = plt.figure(constrained_layout=True)
	plots = figure.add_gridspec(1, 2)

	hyperparameters = [
		{
			"number" : 0,
			"name": "K",
			"dims":plots[0, 0:2],
			"x_label": "K values",
			"x_content" :[i for i in range (2, 21)],
			"plot_title": "Accuracy change with different k values",
			"values": [
				[i, "euclidean"] for i in range (2, 21)
			],
			"stats" : []
		}	
	]
	bestValues = [0]
	for hyperparameter in hyperparameters:
		print("Tuning the ",hyperparameter["name"], " hyperparameter")
		maxAccuracy = 0
		for value in hyperparameter["values"]:
			classifier = KNN(value[0], value[1])
			print("Training...")
			classifier.fit(training_data=trainingData, training_labels=trainingLabels)

			print("Validating...")
			guesses = classifier.test(testing_data=validationData)

			accuracy = calcAccuracy(guesses, validationLabels)
			hyperparameter["stats"].append (round(accuracy, 1))
			if (accuracy > maxAccuracy):
				maxAccuracy = accuracy
				bestValues[hyperparameter["number"]] = value[hyperparameter["number"]]

		f = figure.add_subplot(hyperparameter["dims"])
		if (hyperparameter["number"] == 1):
			bars = f.bar(hyperparameter["x_content"], hyperparameter["stats"])
			f.bar_label(bars)
		else:
			x = hyperparameter["x_content"]
			y = hyperparameter["stats"]
			f.plot(x,y)
			f.set_xticks(hyperparameter["x_content"])
			f.set_yticks(list(range(0,100,10)))
			for index in range(len(x)):
				f.text(x[index], y[index], y[index], size=10)

		f.set_title(hyperparameter["plot_title"])
		f.set_xlabel(hyperparameter["x_label"])
		f.set_ylabel("Accuracy")
	plt.show()

	print ("---------------------------------------")
	print ("Best values for each hyperparameter: ")
	
	print("K ", bestValues[0])
	tuned_params = bestValues

	print("Testing with the tuned hyperparameters...")
	guesses = classify_with_tuned_params(trainingData, trainingLabels, testingData, tuned_params=tuned_params)
	calcAccuracy(guesses=guesses, correctLabels=testingLabels)
	
max_words = 5000
max_len = 50

dataset = First_Dataset()
dataset.preprocess()
dataset.encode_labels()
dataset.embed_tokens(max_words,max_len)
dataset.split()

Training_pad, Validation_pad, Testing_pad, Y_train, Y_val, Y_test = dataset.getData()

# print (Training_pad.shape)
runKNN (Training_pad, Validation_pad, Testing_pad, Y_train, Y_val, Y_test)