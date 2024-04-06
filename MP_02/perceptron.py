from itertools import combinations

import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


class Perceptron:
    def __init__(self, learning_rate, n_iterations, label, second_label=None):
        self.first_label = label
        self.second_label = second_label
        self.weights = None
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def train(self, training_data, labels):
        self.weights = np.zeros(len(training_data[0]) + 1)
        for _ in range(self.n_iterations):
            for x, label in zip(training_data, labels):
                prediction = self.predict(x)
                error = label - prediction
                self.weights[1:] += self.learning_rate * error * x
                self.weights[0] += self.learning_rate * error

    def predict(self, data_i):
        activation = np.dot(data_i, self.weights[1:]) + self.weights[0]
        return 1 if activation > 0 else 0


def main():
    # Generowanie i podział losowych danych
    np.random.seed(0)
    size_of_data = 50
    n_classes = 4
    random_data = np.vstack([
        np.random.normal(loc=[1, 1], scale=[1, 1], size=(size_of_data, 2)),
        np.random.normal(loc=[15, 3], scale=[1, 2], size=(size_of_data, 2)),
        np.random.normal(loc=[3, 15], scale=[2, 2], size=(size_of_data, 2)),
        np.random.normal(loc=[15, 15], scale=[1, 3], size=(size_of_data, 2))
    ])
    labels = np.array([x for x in range(n_classes) for _ in range(size_of_data)])
    unique_labels = set(labels)
    data_train, data_test, labels_train, labels_test = train_test_split(random_data, labels, test_size=0.2,
                                                                        random_state=1)

    # Trening ovr
    learning_rate = 0.1
    n_iterations = 50
    perceptrons_ovr = []
    for value in unique_labels:
        perceptron = Perceptron(learning_rate, n_iterations, value)
        binary_labels = [1 if label == value else 0 for label in labels_train]
        perceptron.train(data_train, binary_labels)
        perceptrons_ovr.append(perceptron)
    # Testowanie ovr
    correct = 0
    total = len(data_test)
    for x, label in zip(data_test, labels_test):
        predictions = {perceptron.first_label: perceptron.predict(x) for perceptron in perceptrons_ovr}
        max_result = max(predictions, key=predictions.get)

        if max_result == label:
            correct += 1
    print(f"OvR accuracy: {correct / total}")
    # Trening ovo
    learning_rate = 0.1
    n_iterations = 50
    perceptrons_ovo = []
    pairs = list(combinations(unique_labels, 2))
    for pair in pairs:
        pair_data_train, pair_labels_train = [], []
        for x, label in zip(data_test, labels_test):
            if label in pair:
                pair_data_train.append(x)
                pair_labels_train.append(1 if label == pair[0] else 0)
        perceptron = Perceptron(learning_rate, n_iterations, pair[0], pair[1])
        perceptron.train(pair_data_train, pair_labels_train)
        perceptrons_ovo.append(perceptron)
    # Testowanie ovo
    correct = 0
    total = len(data_test)
    for x, label in zip(data_test, labels_test):
        points = [0] * n_classes
        for perceptron in perceptrons_ovo:
            prediction = perceptron.predict(x)
            if prediction == 1:
                points[perceptron.first_label] += 1
            else:
                points[perceptron.second_label] += 1
        max_result = np.argmax(points)

        if max_result == label:
            correct += 1

    print(f"OvO accuracy: {correct / total}")

    # Rysowanie danych testowych i treningowych
    colors = ['red', 'blue', 'green', 'purple']
    for _ in range(len(data_test)):
        plt.scatter(data_test[_, 0], data_test[_, 1], c=colors[labels_test[_]], marker='x')
    for _ in range(len(data_train)):
        plt.scatter(data_train[_, 0], data_train[_, 1], c=colors[labels_train[_]], marker='o')

    # Rysowanie granic decyzyjnych
    min_x = np.min(random_data[:, 0])
    max_x = np.max(random_data[:, 0])
    min_y = np.min(random_data[:, 1])
    max_y = np.max(random_data[:, 1])
    for _ in range(len(perceptrons_ovr)):
        [c, a, b] = perceptrons_ovr[_].weights
        x_range = np.array([min_x, max_x])
        y_range = (-a * x_range - c) / b
        plt.plot(x_range, y_range, color=colors[_], linestyle='dotted')

    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    # Rysowanie wykresu
    plt.show()


main()
