# Importowanie potrzebnych bibliotek i modułów
import numpy as np  # Importuje bibliotekę NumPy
import matplotlib.pyplot as plt  # Importuje moduł pyplot z biblioteki Matplotlib
from sklearn.datasets import make_moons  # Importuje funkcję make_moons z scikit-learn
from sklearn.model_selection import train_test_split  # Importuje funkcję train_test_split z scikit-learn
from sklearn.linear_model import LogisticRegression  # Importuje klasę LogisticRegression z scikit-learn
from sklearn.svm import SVC  # Importuje klasę SVC (Support Vector Classifier) z scikit-learn
from sklearn.metrics import accuracy_score  # Importuje funkcję accuracy_score z scikit-learn

# Krok 1: Tworzenie zbioru danych
X, y = make_moons(n_samples=10000, noise=0.3, random_state=42)  # Generuje zbiór danych za pomocą funkcji make_moons

# Krok 2: Podział danych na zbiory uczący i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Dzieli zbiór danych na zbiory uczący i testowy

# Krok 3: Trenowanie klasyfikatora regresji logistycznej
log_reg = LogisticRegression(random_state=42)
'''do uzupełnienia'''  # Trenuje klasyfikator na danych treningowych
log_reg.fit(X_train, y_train)
'''do uzupełnienia'''  # Dokonuje predykcji na danych treningowych
log_reg_y_pred_train = log_reg.predict(X_train)
'''do uzupełnienia'''  # Dokonuje predykcji na danych testowych
log_reg_y_pred_test = log_reg.predict(X_test)
'''do uzupełnienia'''  # Oblicza dokładność klasyfikacji na danych treningowych
log_reg_train_accuracy = accuracy_score(y_train, log_reg_y_pred_train)
'''do uzupełnienia'''  # Oblicza dokładność klasyfikacji na danych testowych
log_reg_test_accuracy = accuracy_score(y_test, log_reg_y_pred_test)
# Krok 4: Trenowanie klasyfikatora SVM default
svm_clf = SVC(random_state=42)
'''do uzupełnienia'''  # Trenuje klasyfikator na danych treningowych
svm_clf.fit(X_train, y_train)
'''do uzupełnienia'''  # Dokonuje predykcji na danych treningowych
svm_y_pred_train = svm_clf.predict(X_train)
'''do uzupełnienia'''  # Dokonuje predykcji na danych testowych
svm_y_pred_test = svm_clf.predict(X_test)
'''do uzupełnienia'''  # Oblicza dokładność klasyfikacji na danych treningowych
svm_train_accuracy = accuracy_score(y_train, svm_y_pred_train)
'''do uzupełnienia'''  # Oblicza dokładność klasyfikacji na danych testowych
svm_test_accuracy = accuracy_score(y_test, svm_y_pred_test)

# Krok 5: Wykresy
# Tworzenie subplotów
fig, axes = plt.subplots(2, 1, figsize=(8, 12))  # Tworzy dwa subploty w układzie pionowym

# Rysowanie granicy decyzyjnej dla regresji logistycznej
ax = axes[0]  # Pierwszy subplot
ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='.', label='Train')  # Rysuje punkty danych treningowych
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='x', label='Test')  # Rysuje punkty danych testowych
xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100),
                     np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100))  # Tworzy siatkę punktów dla wykresu konturu decyzji
Z = log_reg.predict(np.c_[xx.ravel(), yy.ravel()])  # Dokonuje predykcji na punktach siatki
Z = Z.reshape(xx.shape)  # Zmienia kształt wyniku predykcji na kształt siatki
ax.contourf(xx, yy, Z, alpha=0.3)  # Rysuje kontur decyzji
ax.set_title(f"Logistic Regression\nTrain accuracy: {log_reg_train_accuracy:.4f}, Test accuracy: {log_reg_test_accuracy:.4f}")  # Ustawia tytuł wykresu

# Rysowanie granicy decyzyjnej dla SVM
ax = axes[1]  # Drugi subplot
ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='.', label='Train')  # Rysuje punkty danych treningowych
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='x', label='Test')  # Rysuje punkty danych testowych
Z = svm_clf.predict(np.c_[xx.ravel(), yy.ravel()])  # Dokonuje predykcji na punktach siatki
Z = Z.reshape(xx.shape)  # Zmienia kształt wyniku predykcji na kształt siatki
ax.contourf(xx, yy, Z, alpha=0.3)  # Rysuje kontur decyzji
ax.set_title(f"SVM\nTrain accuracy: {svm_train_accuracy:.4f}, Test accuracy: {svm_test_accuracy:.4f}")  # Ustawia tytuł wykresu

plt.tight_layout()  # Dostosowuje układ subplotów
plt.legend()  # Dodaje legendę
plt.savefig("02_log_reg_svm.png")  # Zapisuje wykres do pliku
plt.show()  # Wyświetla wykres

"""
wybrane parametry SVC:
C (penalty parameter): Określa wagę kary za błędy klasyfikacji na danych treningowych.
Domyślna wartość to 1. Wyższe wartości C mogą prowadzić do bardziej skomplikowanego modelu,
który dopasowuje się lepiej do danych treningowych, ale może prowadzić do overfittingu.

Kernel (jądro): Określa typ funkcji jądra używanej w algorytmie SVM.
Dostępne opcje to 'linear', 'poly', 'rbf', 'sigmoid', itp.
Wybór jądra może mieć wpływ na zdolność modelu do dopasowania się do danych.

gamma: Parametr dla 'rbf', 'poly' i 'sigmoid'. Określa wpływ pojedynczego punktu treningowego.
Niskie wartości oznaczają duży zasięg, co oznacza, że punkty są oddalane dalej od siebie,
a wysokie wartości oznaczają bliskie związki punktów treningowych.

degree for 'poly': Stopień wielomianu dla 'poly'. Domyślnie to 3.

class_weight: Określa wagi klas, które są używane podczas uczenia.
Może być przydatny, gdy mamy do czynienia z niezbalansowanymi klasami.
"""