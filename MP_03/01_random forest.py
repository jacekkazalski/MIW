# Importowanie potrzebnych bibliotek i modułów
import numpy as np  # Importuje bibliotekę NumPy
import matplotlib.pyplot as plt  # Importuje moduł pyplot z biblioteki Matplotlib
from sklearn.datasets import make_moons  # Importuje funkcję make_moons z scikit-learn
from sklearn.model_selection import train_test_split  # Importuje funkcję train_test_split z scikit-learn
from sklearn.ensemble import RandomForestClassifier  # Importuje klasę RandomForestClassifier z scikit-learn
from sklearn.metrics import accuracy_score  # Importuje funkcję accuracy_score z scikit-learn

# Krok 1: Tworzenie zbioru danych
X, y = make_moons(n_samples=10000, noise=0.3, random_state=42)  # Generuje zbiór danych za pomocą funkcji make_moons

# Krok 2: Podział danych na zbiory uczący i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Dzieli zbiór danych na zbiory uczący i testowy

# Krok 3: Wykorzystanie lasu losowego jako klasyfikatora
criteria = ['gini', 'entropy']  # Kryteria dla lasu losowego
table_estimators = [5, 20, 100, 500]  # Różne liczby estymatorów w lesie losowym

# Tworzenie subplotów przed iteracją
fig, axes = plt.subplots(4, 2, figsize=(12, 16))  # Tworzy subploty do wyświetlenia wyników dla różnych kombinacji parametrów

for i, criterion in enumerate(criteria):  # Pętla po kryteriach dla lasu losowego
    for j, n_estimator in enumerate(table_estimators):  # Pętla po liczbie estymatorów w lasie losowym
        print(i,j)
        ax = axes[j, i]  # Aktualnie analizowany subplot
        clf = RandomForestClassifier(n_estimators=n_estimator, criterion=criterion, bootstrap=True, max_depth=5, random_state=42)  # Inicjalizuje klasyfikator lasu losowego
        clf.fit(X_train, y_train)  # Dopasowuje model do danych treningowych
        y_pred_train = clf.predict(X_train)# Dokonuje predykcji na danych treningowych
        y_pred_test = clf.predict(X_test)# Dokonuje predykcji na danych testowych
        train_accuracy = accuracy_score(y_train, y_pred_train)# Oblicza dokładność na danych treningowych
        test_accuracy = accuracy_score(y_test, y_pred_test)# Oblicza dokładność na danych testowych
        
        # Tworzenie siatki dla wykresu konturu decyzji
        xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100),
                             np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100))  # Tworzy siatkę punktów dla wykresu
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])  # Dokonuje predykcji dla każdego punktu siatki
        Z = Z.reshape(xx.shape)  # Zmienia kształt wyniku predykcji na kształt siatki
        
        # Rysowanie punktów danych
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='.', label='Train')  # Rysuje punkty danych treningowych
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='x', label='Test')  # Rysuje punkty danych testowych

        # Rysowanie granicy decyzyjnej
        ax.contourf(xx, yy, Z, alpha=0.3)  # Rysuje kontur decyzji na podstawie predykcji klasyfikatora
        
        # Ustawienia wykresu
        ax.set_xlabel('Feature 1')  # Ustawia etykietę dla osi x
        ax.set_ylabel('Feature 2')  # Ustawia etykietę dla osi y
        ax.set_xlim(xx.min(), xx.max())  # Ustawia zakres osi x
        ax.set_ylim(yy.min(), yy.max())  # Ustawia zakres osi y
        ax.set_title(f"Crit:{criterion}, n trees:{n_estimator}\nTrain acc:{train_accuracy:.4f}, Test acc:{test_accuracy:.4f}")  # Ustawia tytuł wykresu z informacją o kryterium, liczbie estymatorów i dokładności trenowania oraz testowania

plt.tight_layout()  # Dopasowuje wykresy do obszaru
plt.legend()  # Dodaje legendę
plt.savefig("01_random_forest.png")  # Zapisuje wykres do pliku
plt.show()  # Wyświetla wykres
