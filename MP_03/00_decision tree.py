# Importowanie niezbędnych bibliotek i modułów
import numpy as np  # Importuje bibliotekę NumPy
import matplotlib.pyplot as plt  # Importuje moduł pyplot z biblioteki Matplotlib
from sklearn.datasets import make_moons  # Importuje funkcję make_moons z scikit-learn
from sklearn.model_selection import train_test_split  # Importuje funkcję train_test_split z scikit-learn
from sklearn.tree import DecisionTreeClassifier  # Importuje klasę DecisionTreeClassifier z scikit-learn
from sklearn.metrics import accuracy_score  # Importuje funkcję accuracy_score z scikit-learn

# Tworzenie zbioru danych
X, y = make_moons(n_samples=10000, noise=0.3,
                  random_state=42)  # Generuje zbiór danych z dwoma przeciwstawianymi kształtami

# Podział danych na zbiory uczący i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)  # Dzieli zbiór danych na zbiory uczący i testowy

# Wykorzystanie drzewa decyzyjnego jako klasyfikatora
criteria = ['gini', 'entropy']  # Kryteria dla drzewa decyzyjnego
max_depths = [3, 5, 10, None]  # Różne maksymalne głębokości drzewa

# Tworzenie subplotów przed iteracją
fig, axes = plt.subplots(4, 2, figsize=(
    12, 16))  # Tworzy subploty dla różnych kombinacji kryteriów i maksymalnych głębokości drzewa

for i, criterion in enumerate(criteria):  # Pętla po kryteriach
    for j, max_depth in enumerate(max_depths):  # Pętla po maksymalnych głębokościach
        ax = axes[j, i]  # Aktualnie analizowany subplot
        '''do uzupełnienia'''  # Tworzy instancję klasyfikatora drzewa decyzyjnego
        clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=42)
        '''do uzupełnienia'''  # Trenuje klasyfikator na danych treningowych
        clf.fit(X_train, y_train)
        '''do uzupełnienia'''  # Dokonuje predykcji na danych treningowych
        y_pred_train = clf.predict(X_train)
        '''do uzupełnienia'''  # Dokonuje predykcji na danych testowych
        y_pred_test = clf.predict(X_test)
        '''do uzupełnienia'''  # Oblicza dokładność klasyfikacji na danych treningowych
        train_accuracy = accuracy_score(y_train, y_pred_train)
        '''do uzupełnienia'''  # Oblicza dokładność klasyfikacji na danych testowych
        test_accuracy = accuracy_score(y_test, y_pred_test)

        # Tworzenie siatki dla wykresu konturu decyzji
        xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100),
                             np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1,
                                         100))  # Tworzy siatkę punktów dla wykresu
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])  # Dokonuje predykcji dla każdego punktu siatki
        Z = Z.reshape(xx.shape)  # Zmienia kształt wyniku predykcji na kształt siatki

        # Rysowanie punktów danych
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='.',
                   label='Train')  # Rysuje punkty danych treningowych
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='x', label='Test')  # Rysuje punkty danych testowych

        # Rysowanie granicy decyzyjnej
        ax.contourf(xx, yy, Z, alpha=0.3)  # Rysuje kontur decyzji na podstawie predykcji klasyfikatora

        # Ustawienia wykresu
        ax.set_xlabel('Feature 1')  # Ustawia etykietę dla osi x
        ax.set_ylabel('Feature 2')  # Ustawia etykietę dla osi y
        ax.set_xlim(xx.min(), xx.max())  # Ustawia zakres osi x
        ax.set_ylim(yy.min(), yy.max())  # Ustawia zakres osi y
        ax.set_title(
            f"Criterion: {criterion}, Max depth: {max_depth}\nTrain accuracy: {train_accuracy:.4f}, Test accuracy: {test_accuracy:.4f}")  # Ustawia tytuł wykresu z informacją o kryterium, maksymalnej głębokości oraz dokładności trenowania i testowania

plt.tight_layout()  # Dopasowuje wykresy do obszaru
plt.legend()  # Dodaje legendę
plt.savefig("00_dec_tree.png")  # Zapisuje wykres do pliku
plt.show()  # Wyświetla wykres

# Na przykład, jeśli użyjemy plt.subplots(2, 2), aby utworzyć siatkę 2x2 podwykresów, tablica axes będzie miała kształt (2, 2). Aby odwołać się do konkretnego podwykresu w tej tablicy, używamy indeksów.
