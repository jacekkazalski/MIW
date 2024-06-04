from keras.src.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from keras.src.models import Model
from keras.src.layers import Input, Dense
from keras.src.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

xx = 9
# Wczytywanie danych
with open(f"Dane/dane{xx}.txt", "r") as file:
    data = file.readlines()

x, y = zip(*[map(float, line.split()) for line in data])
# Normalizacja danych
scaler = MinMaxScaler()
x = scaler.fit_transform(np.array(x).reshape(-1, 1))
y = scaler.fit_transform(np.array(y).reshape(-1, 1))
# Podział na zbiory testowe i treningowe
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
# Parametry
learning_rate = 0.004
epochs = 300
hidden_size = 32
# Budowanie modelu
inputs = Input(shape=(1,))
x = Dense(hidden_size, activation='relu')(inputs)
x = Dense(hidden_size, activation='relu')(x)
outputs = Dense(1)(x)
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error',
              metrics=['mean_absolute_error', "r2_score", "mean_squared_error"])
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(x_train, y_train, epochs=epochs, batch_size=32, validation_data=(x_test, y_test), verbose=0)
result = model.evaluate(x_test, y_test, verbose=0)
print(f"Result for {xx}: {result}")

# Wykresy
plt.rcParams.update({'font.size': 16})
plt.figure(figsize=(15, 10))

# Wykres MSE vs. epoki
plt.subplot(2, 2, 1)
plt.plot(range(len(history.history['loss'])), history.history['loss'], marker='.')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.title(f'MSE vs. Epochs, Final MSE: {round(history.history["loss"][-1], 3)}')
plt.ylim(0, 0.2)

# Wykres R^2 vs. epoki
plt.subplot(2, 2, 2)
plt.plot(range(len(history.history['r2_score'])), history.history['r2_score'], marker='.')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.title(f'R^2 vs. Epochs, Final R^2: {round(history.history["r2_score"][-1], 2)}')
plt.ylim(0, 1)
plt.tight_layout()

# Wykres punktowy dla zbioru treningowego i testowego oraz predykcji
plt.subplot(2, 2, 3)
plt.scatter(x_train, y_train, color='blue', label='Training Data')
plt.scatter(x_test, y_test, color='red', label='Test Data')
plt.scatter(x_train, y_train, color='green', label='Train Predictions', marker='s')
plt.scatter(x_test, y_test, color='green', label='Test Predictions', marker='x')
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Training and Test Data with Predictions')
plt.legend()
plt.tight_layout()

plt.subplot(2,2,4)
plt.axis('off')
plt.text(0, 0.5, f"DataSet - {xx}\nNeurons - {hidden_size}\nEpochs={epochs}\nLearning Rate={learning_rate}")


plt.show()
