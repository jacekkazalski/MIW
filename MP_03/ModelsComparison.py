import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

# Wybór danych
data_n1 = 10
data_n2 = 15
for data_n in [data_n1, data_n2]:
    # Wczytywanie danych
    with open(f"./Dane/dane{data_n}.txt") as file:
        data = file.readlines()
    # Rozdzielanie danych
    x_data = []
    y_data = []
    for line in data:
        line_arr = line.split(' ')
        x_data.append(float(line_arr[0]))
        y_data.append(float(line_arr[1]))
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=0)
    # Przekształcanie do postaci kolumnowej
    x_train = np.array(x_train).reshape(-1, 1)
    y_train = np.array(y_train)
    x_test = np.array(x_test).reshape(-1, 1)
    y_test = np.array(y_test)
    # Model liniowy
    lin_reg = LinearRegression()
    lin_reg.fit(x_train, y_train)
    y_pred_lin = lin_reg.predict(x_test)
    a = lin_reg.coef_
    b = lin_reg.intercept_
    precision_lin_reg = r2_score(y_test, y_pred_lin)
    # Model wielomianowy
    poly_features = PolynomialFeatures(degree=2)
    x_train_poly = poly_features.fit_transform(x_train)
    x_test_poly = poly_features.transform(x_test)
    model_poly = LinearRegression()
    model_poly.fit(x_train_poly, y_train)
    y_pred_poly = model_poly.predict(x_test_poly)
    precision_poly = r2_score(y_test, y_pred_poly)
    print(f"Precision dane{data_n} {precision_lin_reg} {precision_poly}")
    # Wykres prezentujący punkty z danych i dopasowane modele
    plt.scatter(x_train, y_train, color='blue', label='Dane treningowe')
    plt.scatter(x_test, y_test, color='green', label='Dane testowe')
    x_range = np.linspace(min(x_train), max(x_train), 100).reshape(-1, 1)
    plt.plot(x_range, lin_reg.predict(x_range), color='red',
             label=f'y = {a[0]:.2f}x + {b:.2f}, r^2={precision_lin_reg:.2f}')
    plt.plot(x_range, model_poly.predict(poly_features.transform(x_range)), color='orange',
             label=f'y = {model_poly.intercept_:.2f} + {model_poly.coef_[1]:.2f}x + {model_poly.coef_[2]:.2f}x^2, r^2={precision_poly:.2f}')
    plt.xlabel('Wartość X')
    plt.ylabel('Wartość Y')
    plt.title(f'Porównanie modeli dane{data_n}')
    plt.legend()
    plt.savefig(f'porownanie_modeli{data_n}.png')
    plt.show()
