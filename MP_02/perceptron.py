import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


class Perceptron:
    def __init__(self, learning_rate, n_iterations):
        self.this_label = None
        self.weights = None
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def train_ovr(self, training_data, labels, this_label):
        self.weights = np.zeros(len(training_data[0]) + 1)
        self.this_label = this_label
        binary_labels = [1 if label == this_label else 0 for label in labels]
        for _ in range(self.n_iterations):
            for x, label in zip(training_data, binary_labels):
                prediction = self.predict(x)
                error = label - prediction
                self.weights[1:] += self.learning_rate * error * x
                self.weights[0] += self.learning_rate * error

    def predict(self, data_i):
        activation = np.dot(data_i, self.weights[1:]) + self.weights[0]
        return 1 if activation > 0 else 0


def main():
    iris = datasets.load_iris()
    data = iris.data
    labels = iris.target
    unique_labels = set(labels)
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2, random_state=1)
    learning_rate = 0.1
    n_iterations = 50
    perceptrons = []

    # Trening
    for value in unique_labels:
        perceptron = Perceptron(learning_rate, n_iterations)
        perceptron.train_ovr(data_train, labels_train, value)
        perceptrons.append(perceptron)

    # Testowanie
    correct = 0
    total = len(data_test)
    for x, label in zip(data_test, labels_test):
        predictions = {perceptron.this_label: perceptron.predict(x) for perceptron in perceptrons}
        max_result = max(predictions, key=predictions.get)

        if max_result == label:
            correct += 1

    print(f"Accuracy: {correct / total}")


main()
