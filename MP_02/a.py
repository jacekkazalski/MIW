import numpy as np  # Importuje bibliotekę do pracy na tablicach i macierzach
import matplotlib.pyplot as plt  # Importuje bibliotekę do tworzenia wykresów
from sklearn.datasets import make_blobs  # Importuje funkcję do generowania danych
from sklearn.model_selection import train_test_split  # Importuje funkcję do podziału danych na zbiór treningowy i testowy
from sklearn.preprocessing import StandardScaler  # Importuje funkcję do standaryzacji danych

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate  # Ustawia współczynnik uczenia
        self.n_iterations = n_iterations  # Ustawia liczbę iteracji

    def fit(self, X, y):
        n_samples, n_features = X.shape  # Pobiera liczbę próbek i cech
        self.weights = np.zeros(n_features)  # Inicjalizuje wagi na zero
        self.bias = 0  # Inicjalizuje obciążenie na zero

        for _ in range(self.n_iterations):  # Pętla ucząca
            linear_model = np.dot(X, self.weights) + self.bias  # Oblicza model liniowy
            y_predicted = self.sigmoid(linear_model)  # Przewiduje wartości za pomocą funkcji sigmoidalnej

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))  # Oblicza gradient wag
            db = (1 / n_samples) * np.sum(y_predicted - y)  # Oblicza gradient obciążenia

            self.weights -= self.learning_rate * dw  # Aktualizuje wagi
            self.bias -= self.learning_rate * db  # Aktualizuje obciążenie

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias  # Oblicza model liniowy
        y_predicted = self.sigmoid(linear_model)  # Przewiduje wartości za pomocą funkcji sigmoidalnej
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]  # Dokonuje klasyfikacji na podstawie progowania
        return y_predicted_cls  # Zwraca przewidywane klasy

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))  # Implementacja funkcji sigmoidalnej


def calculate_accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)  # Oblicza dokładność klasyfikacji
    return accuracy  # Zwraca dokładność

# Generowanie danych
X, y = make_blobs(n_samples=50, centers=2, n_features=2, random_state=424)  # Tworzy zestaw danych
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=424)  # Dzieli dane na zbiory treningowy i testowy

# Standaryzacja danych
scaler = StandardScaler()  # Inicjalizuje obiekt do standaryzacji
X_train = scaler.fit_transform(X_train)  # Standaryzuje zbiór treningowy
X_test = scaler.transform(X_test)  # Standaryzuje zbiór testowy
'''
Standaryzacja danych jest procesem przekształcania wartości cech w taki sposób,
aby miały średnią równą zero i odchylenie standardowe równa jeden.
Polega to na odejmowaniu średniej wartości danej cechy od każdej obserwacji,
a następnie dzieleniu wyniku przez odchylenie standardowe tej cechy.
Ułatwienie interpretacji,Uniezależnienie od skali,Poprawa wydajności modelu,Zmniejszenie wpływu wartości odstających.
'''

# Inicjalizacja i trenowanie modelu
log_reg = LogisticRegression(learning_rate=0.1, n_iterations=100)  # Inicjalizuje model regresji logistycznej
log_reg.fit(X_train, y_train)  # Trenuje model na danych treningowych

# Przewidywanie klas dla danych testowych
y_pred = log_reg.predict(X_test)  # Przewiduje klasy dla danych testowych

# Obliczanie dokładności i wyświetlanie wyników
accuracy = calculate_accuracy(y_test, y_pred)  # Oblicza dokładność modelu
print(f'Dokładność regresji logistycznej: {accuracy * 100:.2f}%')  # Wyświetla dokładność modelu

# Wykres danych treningowych i testowych oraz przewidywanych klas przez regresję logistyczną
plt.figure(figsize=(10, 6))  # Tworzy nowy wykres

# Dane treningowe
plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], color='red', marker='o', label='Class 0 (Train)')  # Wyświetla punkty klasy 0 ze zbioru treningowego
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='blue', marker='o', label='Class 1 (Train)')  # Wyświetla punkty klasy 1 ze zbioru treningowego

# Dane testowe
plt.scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1], color='red', marker='x', label='Class 0 (Test)')  # Wyświetla punkty klasy 0 ze zbioru testowego
plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='blue', marker='x', label='Class 1 (Test)')  # Wyświetla punkty klasy 1 ze zbioru testowego

# Przewidywane klasy przez regresję logistyczną
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1  # Określa zakres na osi x
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1  # Określa zakres na osi y
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))  # Tworzy siatkę punktów
Z = log_reg.predict(np.c_[xx.ravel(), yy.ravel()])  # Przewiduje klasy dla każdego punktu na siatce
Z = np.array(Z).reshape(xx.shape)  # Zmienia kształt przewidywań
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)  # Tworzy wykres konturowy dla przewidywanych klas

plt.title(f'Dokładność regresji logistycznej: {accuracy * 100:.2f}%')  # Dodaje tytuł wykresu
plt.xlabel('Feature 1')  # Dodaje opis osi x
plt.ylabel('Feature 2')  # Dodaje opis osi y
plt.legend()  # Dodaje legendę
plt.show()  # Wyświetla wykres