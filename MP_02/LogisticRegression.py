import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


# Zwraca wynik funkcji sigmoidalnej 1/(1+e^-x) = <0;1>
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LogisticRegression:

    def __init__(self, learning_rate=0.001, n_iter=1000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    def train(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.n_iter):
            linear_predictions = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_predictions)

            weight_gradient = (1 / n_samples) * np.dot(X.T, (predictions - y))
            bias_gradient = (1 / n_samples) * np.sum(predictions - y)
            print(weight_gradient, bias_gradient)
            self.weights -= self.learning_rate * weight_gradient
            self.bias -= self.learning_rate * bias_gradient

    def predict(self, X):
        linear_predictions = np.dot(X, self.weights) + self.bias
        predictions = sigmoid(linear_predictions)
        return predictions


def main():
    size_of_data = 200
    n_classes = 4
    n_features = 2

    # Generowanie danych
    data, labels = make_classification(
        n_samples=size_of_data,
        n_features=n_features,
        n_informative=2,
        n_clusters_per_class=1,
        n_redundant=0,
        n_classes=n_classes,
        random_state=1
    )

    # Podział danych
    data_train, data_test, label_train, label_test = train_test_split(data, labels, random_state=0)
    models = []

    # Tworzenie modelu dla każdej klasy
    for _ in range(n_classes):
        model = LogisticRegression()
        binary_labels = np.where(label_train == _,1,0)
        print(binary_labels)
        model.train(data_train, binary_labels)
        models.append(model)

    correct = 0
    total = len(data_test)
    predictions = np.zeros(len(data_test))
    predicted_labels = np.zeros(len(data_test))
    for _ in range(len(models)):
        predictions_tmp = models[_].predict(data_test)

        for i in range(len(predictions_tmp)):
            print(f"Model {_} predictions: {predictions_tmp[i]} Highest: {predictions[i]}")
            if predictions_tmp[i] > predictions[i]:
                predictions[i] = predictions_tmp[i]
                predicted_labels[i] = _

    for i in range(len(predicted_labels)):
        print(f" Prediction: {predicted_labels[i]} Actual: {label_test[i]}")
        if predicted_labels[i] == label_test[i]:
            correct += 1
    print(f"Logistic Regression Accuracy: {correct / total}")


main()
