import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

import keras
from keras import layers
from keras.src.models import Model
from keras.src.layers import Input, Dense
from keras.src.optimizers import Adam

# Tworzenie zbioru danych
X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)
# Podział danych na zbiory uczący i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Parametry
learning_rate = 0.005
epochs = 100
hidden_size = 32
#Budowanie modelu
inputs = Input(shape=X_train.shape[1:])
x= Dense(hidden_size, activation='relu')(inputs)
x = Dense(hidden_size, activation='relu')(x)
outputs = Dense(1, activation='sigmoid')(x)
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy',
              metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), verbose=0)

_, test_accuracy = model.evaluate(X_test, y_test)
_, train_accuracy = model.evaluate(X_train, y_train)
# Rysowanie granicy decyzyjnej dla sieci neuronowej
plt.figure(figsize=(15, 10))

# Wykres danych i granicy decyzyjnej
plt.subplot(2, 2, 3)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='.', label='Train')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='x', label='Test')

xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100),
                     np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100))

grid = np.c_[xx.ravel(), yy.ravel()]
Z = model.predict(grid)
Z = np.round(Z).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.title(f"Neural Network\nTest accuracy: {test_accuracy:.3f} and train accuracy: {train_accuracy:.3f}"
          f"\nEpochs: {epochs} Learning Rate: {learning_rate} Neurons: {hidden_size}")
plt.legend()

# Wykresy accuracy vs epoki
plt.subplot(2, 2, 2)
epochs_range = range(epochs)
plt.plot(epochs_range, history.history['accuracy'], label='Train Accuracy', color='green')
plt.plot(epochs_range, history.history['val_accuracy'], label='Val Accuracy', color='blue')
plt.xlabel('Epochs')
plt.legend()
plt.ylabel('Accuracy')
plt.title('Accuracy vs epochs')
plt.ylim(min(history.history['accuracy'])-0.01, max(history.history['accuracy'])+0.01)
# Wykres binary_crossentropy vs epoki
plt.subplot(2,2,1)
plt.plot(epochs_range, history.history['loss'], label='binary_crossentropy', color='green')
plt.xlabel('Epochs')
plt.ylabel('Crossentropy')
plt.ylim(min(history.history['loss'])-0.01, max(history.history['loss'])+0.01)

plt.tight_layout()
plt.show()