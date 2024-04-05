import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


class Perceptron:
    def __init__(self, learning_rate, n_iterations):
        self.weights = None
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def train(self, data, labels):
        self.weights = np.zeros(len(data[0]) + 1)
        for _ in range(self.n_iterations):
            for data_i, label in zip(data, labels):
                prediction = self.predict(data_i)
                error = label - prediction
                self.weights[1:] += self.learning_rate * error * data_i
                self.weights[0] += self.learning_rate * error
                # print(f"Prediction: {prediction} \nLabel: {label}\nWeights: {self.weights}")

    def predict(self, data_i):
        activation = np.dot(data_i, self.weights[1:]) + self.weights[0]
        return 1 if activation > 0 else 0

    def accuracy(self, data, labels):
        correct = 0
        for data_i, label in zip(data, labels):
            if self.predict(data_i) == label:
                correct += 1
        total = len(labels)
        return correct / total


def main():
    iris = datasets.load_iris()
    data = iris.data
    labels = iris.target
    # Tablica bool gdzie label < 2
    indices = labels < 2
    data = data[indices]
    labels = labels[indices]

    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2, random_state=1)

    learning_rate = 0.1
    n_iterations = 50
    perceptron = Perceptron(learning_rate, n_iterations)
    perceptron.train(data_train, labels_train)
    accuracy = perceptron.accuracy(data_test, labels_test)
    print(accuracy)


main()
