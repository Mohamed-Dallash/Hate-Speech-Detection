from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from First_Dataset import First_Dataset

def convert_to_tfidf(X_train, X_val, X_test, max_features=5000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)
    return X_train_tfidf, X_val_tfidf, X_test_tfidf

def train_SVM(X_train_tfidf, X_val_tfidf, y_train, kernel='linear', gamma='auto', C=1.0):
    classifier = SVC(kernel=kernel, C=C, gamma = gamma)
    
    print("Training...")
    classifier.fit(X_train_tfidf, y_train)

    print("Validating...")
    predictions = classifier.predict(X_val_tfidf)
    
    return classifier

def evaluate_SVM(classifier, X_test_tfidf):
    predictions = classifier.predict(X_test_tfidf)
    return predictions

def calcAccuracy(predictions, y_test):
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    return accuracy, report

dataset = First_Dataset()
dataset.preprocess()

dataset.split_2()
Training_pad, Validation_pad, Testing_pad, Y_train, Y_val, Y_test = dataset.getData()

X_train_tfidf, X_val_tfidf, X_test_tfidf = convert_to_tfidf(Training_pad, Validation_pad, Testing_pad)

classifier = train_SVM(X_train_tfidf, X_val_tfidf, Y_train)
predictions = evaluate_SVM(classifier, X_test_tfidf)
accuracy, report = calcAccuracy(predictions, Y_test)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)


