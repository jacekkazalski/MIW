import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class NeuralNetworkRegression:
    def __init__(self, input_size, hidden_size, output_size, activation):
        self.activation = activation
        # Inicjalizacja wag i biasów oraz historii wartości MSE i R^2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # Inicjalizacja wag dla warstwy wejściowej-ukrytej i ukrytej-wyjściowej
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        # Inicjalizacja biasów dla warstwy ukrytej i wyjściowej
        self.bias_hidden = np.zeros(hidden_size)
        self.bias_output = np.zeros(output_size)
        # Inicjalizacja historii wartości MSE i R^2
        self.history_mse = []
        self.history_r2 = []

    def relu(self, x):
        # Funkcja aktywacji ReLU
        return np.maximum(0, x)

    def relu_derivative(self, x):
        # Pochodna funkcji aktywacji ReLU
        return np.where(x > 0, 1, np.where(x < 0, 0, 0.5))

    def arc_tang(self, x):
        return np.arctan(x)

    def arc_tang_derivative(self, x):
        return 1 / (1 + np.sqrt(x))

    def forward(self, X):
        # Przekazanie sygnału przez sieć w kierunku przód
        hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        if self.activation == 0:
            hidden_output = self.relu(hidden_input)
        elif self.activation == 1:
            hidden_output = self.arc_tang(hidden_input)
        output_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
        if self.activation == 0:
            return self.relu(output_input)
        elif self.activation == 1:
            return self.arc_tang(output_input)

    def backward(self, X, y, output, learning_rate, reg_lambda):
        # Propagacja wsteczna błędu przez sieć
        output_error = y - output
        hidden_output = self.relu(np.dot(X, self.weights_input_hidden) + self.bias_hidden)
        gradient_hidden_output = np.dot(hidden_output.T, output_error)
        hidden_error = np.dot(output_error, self.weights_hidden_output.T)
        if self.activation == 0:
            hidden_error *= self.relu_derivative(hidden_output)
        elif self.activation == 1:
            hidden_error *= self.arc_tang_derivative(hidden_output)
        gradient_input_hidden = np.dot(X.T, hidden_error)
        # Aktualizacja wag i biasów
        self.weights_hidden_output += (gradient_hidden_output - reg_lambda * self.weights_hidden_output) * learning_rate
        self.weights_input_hidden += (gradient_input_hidden - reg_lambda * self.weights_input_hidden) * learning_rate
        self.bias_output += np.sum(output_error, axis=0) * learning_rate
        self.bias_hidden += np.sum(hidden_error, axis=0) * learning_rate

    def train(self, X, y, epochs, learning_rate, reg_lambda):
        # Trenowanie sieci neuronowej
        print("Start training neural networks")
        for epoch in range(1, epochs):
            output = self.forward(X)
            self.backward(X, y, output, learning_rate, reg_lambda)
            if epoch % (epochs // 10) == 0:
                print(f"[%] {epoch / epochs}")  # Wyświetlenie postępu procesu trenowania
            mse = mean_squared_error(y, output)
            r2 = r2_score(y, output)
            self.history_mse.append(mse)
            self.history_r2.append(r2)
            if r2 >= 0.95:  # Warunek zatrzymania trenowania, jeśli R^2 osiągnie 0,95
                print("Training stopped, R^2 score reached 0.95")
                break


# Pobieranie aktualnej daty i godziny
now = datetime.now()
XX = 7
# Wczytanie danych z pliku
with open(f"Dane/dane{XX}.txt", "r") as file:
    data = file.readlines()

# Przetwarzanie danych wejściowych i wyjściowych
x_data, y_data = zip(*[map(float, line.split()) for line in data])

# Normalizacja danych
scaler = MinMaxScaler()
X_data_normalized = scaler.fit_transform(np.array(x_data).reshape(-1, 1))
y_data_normalized = scaler.fit_transform(np.array(y_data).reshape(-1, 1))

# Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X_data_normalized, y_data_normalized, test_size=0.2,
                                                    random_state=0)

# Inicjalizacja i trenowanie sieci neuronowej
input_size = 1
hidden_size = 1000
output_size = 1
epochs = 30000
learning_rate = 0.001
reg_lambda = 0.04
# 0 = relu, 1 = arc_tan
activation = 0
nn = NeuralNetworkRegression(input_size, hidden_size, output_size, 0)
nn.train(X_train, y_train, epochs=epochs, learning_rate=learning_rate, reg_lambda=reg_lambda)

# Obliczenie przewidywań dla danych treningowych i testowych
output_train = nn.forward(X_train)
output_test = nn.forward(X_test)

# Obliczenie ostatnich wartości MSE i R^2
final_mse = nn.history_mse[-1]
final_r2 = nn.history_r2[-1]

# Wykresy
plt.rcParams.update({'font.size': 16})
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
plt.scatter(range(len(nn.history_mse)), nn.history_mse, marker='.')
plt.xlabel('')
plt.ylabel('MSE')
plt.title(f'MSE vs. Epochs, Final MSE:{round(final_mse, 3)}')
plt.ylim(0, 0.2)

plt.subplot(2, 2, 2)
plt.scatter(range(len(nn.history_r2)), nn.history_r2, marker='.')
plt.xlabel('Epochs')
plt.ylabel('R^2')
plt.title(f'R^2 vs. Epochs, Final R^2:{round(final_r2, 2)}')
plt.ylim(0, 1)
plt.tight_layout()

plt.subplot(2, 2, 3)
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.scatter(X_test, y_test, color='red', label='Test Data')
plt.scatter(X_train, output_train, color='green', label='Predictions', marker='s')
plt.scatter(X_test, output_test, color='green', label='Predictions', marker='x')
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Neurons={hidden_size}, Epochs={len(nn.history_r2)}, learn_r={learning_rate}, lambda={reg_lambda}')
plt.legend()

plt.subplot(2, 2, 4)
plt.axis('off')
if activation == 0:
    act_txt = "relu"
    der_txt = "\n 1 dla x>0\n 0 dla x<0\n 0.5 dla x=0\n"
elif activation == 1:
    act_txt = "arc_tang"
    der_txt = "1/(1+x^2)\n"
plt.text(0, 0.5,
         f"DataSet - {XX}\nNeurony - {hidden_size}\nEpoki - {len(nn.history_r2)}\nFunkcja aktywacji "
         f"-{act_txt}\nPochodna:{der_txt}Inic. wag - rozk. norm.(0,1)\nInic. biasów - 0", fontsize=24,
         verticalalignment='center', horizontalalignment='left')

# Zapisywanie wykresu
nazwa_pliku = now.strftime(f'DataSet_{XX}_%Y%m%d_%H%M%S_neu_{hidden_size}_epch_{len(nn.history_r2)}.png')
plt.savefig(nazwa_pliku)
plt.show()
