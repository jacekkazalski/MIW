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
            self.weights -= self.learning_rate * weight_gradient
            self.bias -= self.learning_rate * bias_gradient

    def predict(self, X):
        linear_predictions = np.dot(X, self.weights) + self.bias
        predictions = sigmoid(linear_predictions)
        return predictions


def softmax(logits):
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


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
        model.train(data_train, binary_labels)
        models.append(model)

    # Przewidywanie klas dla danych testowych i wyświetlanie dokładności wyników
    correct = [0] * 4
    total = [0] * 4
    logits = np.zeros(shape=[len(data_test), n_classes])
    # Liczenie logits
    for _ in range(len(models)):
        logits[:, _] = models[_].predict(data_test)
    # Zamiana na rozkład prawdopodobieństwa softmax
    predictions = softmax(logits)
    # Zliczanie prawidłowych odpowiedzi
    for i, arr in enumerate(predictions):
        print(max(arr))
        total[label_test[i]] += 1
        predicted_label = np.argmax(arr)
        if predicted_label == label_test[i]:
            correct[predicted_label] += 1
    # Wyświetlenie wyników dokładności dla poszczególnych klas
    print(f"Logistic Regression Accuracy: {np.sum(correct) / np.sum(total)}")
    for _ in range(len(correct)):
        print(f"Accuracy for class {_}: {correct[_]} / {total[_]} {correct[_] / total[_]}")
    # Rysowanie wykresu rozkładu prawdopodobieństwa
    plot_labels = ['Class 0', 'Class 1', 'Class 2', 'Class 3']
    mean_prediction = np.mean(predictions, axis=0)
    print(mean_prediction)
    plt.bar(plot_labels, mean_prediction)
    plt.title("Rozkład prawdopodobieństwa między klasami")
    plt.xlabel("Klasa")
    plt.ylabel("Średnie prawdopodobieństwo")
    y_min_r = min(mean_prediction) - 0.01
    y_max_r = max(mean_prediction) + 0.01
    plt.ylim(y_min_r, y_max_r)
    for i in np.arange(0.24, 0.26, 0.0025):
        plt.axhline(y=i, color='r', linestyle='--')
    plt.savefig("probability_dist")
    plt.show()
    # Rysowanie wykresu obszarów decyzyjnych
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

    # Dodanie punktu do legendy dla każdej klasy
    for i in range(len(models)):
        plt.scatter([], [], c=colors[label_test[i]], label=plot_labels[i])
    plt.legend(loc='best')  # Wyświetlenie legendy na wykresie
    for _ in range(len(data_test)):
        plt.scatter(data_test[_, 0], data_test[_, 1], c=colors[label_test[_]], marker='x')
    for _ in range(len(data_train)):
        plt.scatter(data_train[_, 0], data_train[_, 1], c=colors[label_train[_]], marker='o')
    plt.title("Regiony decyzyjne modelu")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.savefig("decision_regions.png")
    plt.show()


main()
