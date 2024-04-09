import numpy as np  # Importuje bibliotekę NumPy pod aliasem np, używana do obliczeń numerycznych.
import matplotlib.pyplot as plt  # Importuje moduł pyplot z biblioteki Matplotlib, służący do rysowania wykresów.
from sklearn.datasets import make_moons  # Importuje funkcję make_moons z scikit-learn, która generuje dane o dwóch przeciwstawnych kształtach.
from sklearn.model_selection import train_test_split  # Importuje funkcję train_test_split z scikit-learn, używaną do podziału danych na zbiory uczący i testowy.
from sklearn.ensemble import RandomForestClassifier  # Importuje klasę RandomForestClassifier z scikit-learn, służącą do klasyfikacji za pomocą lasów losowych.
from sklearn.linear_model import LogisticRegression  # Importuje klasę LogisticRegression z scikit-learn, służącą do klasyfikacji za pomocą regresji logistycznej.
from sklearn.svm import SVC  # Importuje klasę SVC (Support Vector Classifier) z scikit-learn, służącą do klasyfikacji za pomocą maszyny wektorów nośnych (SVM).
from sklearn.ensemble import VotingClassifier  # Importuje klasę VotingClassifier z scikit-learn, służącą do łączenia wielu klasyfikatorów w jedną grupę.
from sklearn.metrics import accuracy_score  # Importuje funkcję accuracy_score z scikit-learn, używaną do obliczania dokładności klasyfikacji.

# Krok 1: Tworzenie zbioru danych
X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)  # Generuje zbiór danych za pomocą funkcji make_moons.

# Krok 2: Podział danych na zbiory uczący i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Dzieli zbiór danych na zbiory uczący i testowy.

# Krok 3: Tworzenie klasyfikatorów
svm_clf = '''do uzupełnienia'''  # Tworzy instancję klasyfikatora SVM.
log_reg_clf = '''do uzupełnienia'''  # Tworzy instancję klasyfikatora regresji logistycznej.
rf_clf = '''do uzupełnienia'''  # Tworzy instancję klasyfikatora lasu losowego.

# Krok 4: Połączenie klasyfikatorów w VotingClassifier
voting_clf = '''do uzupełnienia'''

# Krok 5: Trenowanie modelu VotingClassifier
'''do uzupełnienia'''   # Trenuje model VotingClassifier na danych treningowych.

# Krok 6: Ocena modelu
train_accuracy = '''do uzupełnienia'''   # Oblicza dokładność modelu na danych treningowych.
test_accuracy = '''do uzupełnienia'''   # Oblicza dokładność modelu na danych testowych.

# Krok 7: Rysowanie wyników
plt.figure(figsize=(10, 10))  # Ustawia rozmiar wykresu.
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='.', label='Train')  # Rysuje punkty danych treningowych.
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='x', label='Test')  # Rysuje punkty danych testowych.
xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100),  # Tworzy siatkę dla wykresu konturu decyzji.
                     np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100))
Z = voting_clf.predict(np.c_[xx.ravel(), yy.ravel()])  # Dokonuje predykcji na całej siatce.
Z = Z.reshape(xx.shape)  # Przekształca wynik predykcji do oryginalnego kształtu siatki.
plt.contourf(xx, yy, Z, alpha=0.3)  # Rysuje kontur decyzji na podstawie predykcji VotingClassifier.
plt.xlabel('Feature 1')  # Ustawia etykietę osi x.
plt.ylabel('Feature 2')  # Ustawia etykietę osi y.
plt.title(f"VotingClassifier\nTrain acc:{train_accuracy:.4f}, Test acc:{test_accuracy:.4f}")  # Ustawia tytuł wykresu.
plt.legend()  # Dodaje legendę.
plt.tight_layout()  # Dostosowuje wygląd wykresu.
plt.show()  # Wyświetla wykres.
plt.savefig("03_voting_classifier.png")  # Zapisuje wykres do pliku.