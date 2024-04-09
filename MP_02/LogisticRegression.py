import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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
            # print(weight_gradient, bias_gradient)
            self.weights -= self.learning_rate * weight_gradient
            self.bias -= self.learning_rate * bias_gradient

    def predict(self, X):
        linear_predictions = np.dot(X, self.weights) + self.bias
        predictions = sigmoid(linear_predictions)
        return predictions


# noinspection DuplicatedCode
def main():
    size_of_data = 200
    n_classes = 4
    n_features = 2

    # Generowanie danych
    data, labels = make_blobs(
        n_samples=size_of_data,
        centers=n_classes,
        n_features=n_features,
        random_state=99
    )
    # Standaryzacja danych
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    # Podział danych
    data_train, data_test, label_train, label_test = train_test_split(data, labels, random_state=0)
    models = []

    # Tworzenie modelu dla każdej klasy
    for _ in range(n_classes):
        model = LogisticRegression()
        binary_labels = np.where(label_train == _, 1, 0)
        # print(binary_labels)
        model.train(data_train, binary_labels)
        models.append(model)

    # Przewidywanie klas dla danych testowych i wyświetlanie dokładności wyników
    correct = [0]*4
    total = [0]*4
    print(correct)
    predictions = np.zeros(len(data_test))
    predicted_labels = np.zeros(len(data_test))
    for _ in range(len(models)):
        predictions_tmp = models[_].predict(data_test)

        for i in range(len(predictions_tmp)):
            # print(f"Model {_} predictions: {predictions_tmp[i]} Highest: {predictions[i]}")
            if predictions_tmp[i] > predictions[i]:
                predictions[i] = predictions_tmp[i]
                predicted_labels[i] = _
    # Zliczanie prawidłowych odpowiedzi
    for _ in range(len(predicted_labels)):
        total[int(label_test[_])] += 1
        if predicted_labels[_] == label_test[_]:
            correct[int(predicted_labels[_])] += 1
    print(f"Logistic Regression Accuracy: {np.sum(correct)/ np.sum(total)}")
    for _ in range(len(correct)):
        print(f"Accuracy for class {_}: {correct[_]} / {total[_]} {correct[_] / total[_]}")

    # Rysowanie wykresu
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1  # Określa zakres na osi x
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1  # Określa zakres na osi y
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    Z = np.zeros(xx.shape)  # Macierz przechowująca klasy dla każdego punktu siatki
    colors = ['blue', 'red', 'green', 'orange']
    # Wybór koloru dla każdego z punktów na siatce
    for i in range(len(xx)):
        for j in range(len(yy)):
            point = np.array([xx[i, j], yy[i, j]]).reshape(1, -1)
            probabilities = [model.predict(point)[0] for model in models]
            Z[i, j] = np.argmax(probabilities)  # Wybór klasyfikatora z najwyższym prawdopodobieństwem
    plt.contourf(xx, yy, Z, alpha=0.5, cmap=plt.colormaps['Set3'])

    # Tworzenie etykiet legendy
    legend_labels = ['Class ' + str(i) for i in range(len(models))]
    # Dodanie punktu do legendy dla każdej klasy
    for i in range(len(models)):
        plt.scatter([], [], c=colors[label_test[i]], label=legend_labels[i])
    plt.legend(loc='best')  # Wyświetlenie legendy na wykresie
    for _ in range(len(data_test)):
        plt.scatter(data_test[_, 0], data_test[_, 1], c=colors[label_test[_]], marker='x')
    for _ in range(len(data_train)):
        plt.scatter(data_train[_, 0], data_train[_, 1], c=colors[label_train[_]], marker='o')

    plt.savefig("log_reg.png")


main()
