from public_tests_logistic_multi import compute_cost_test, compute_gradient_test
from utils import load_data_csv_multi_logistic,zscore_normalize_features

from LogisticRegressionMulti import LogisticRegMulti
from LogisticRegressionMulti import cost_test_multi_obj
from LogisticRegressionMulti import compute_gradient_multi_obj
import pandas as pd
import numpy as np


def test_gradient(x_train, y_train):
    print("------------test_gradient-----------")
    compute_gradient_test(compute_gradient_multi_obj)


def test_cost(x_train, y_train):
     print("------------test_cost-----------")
     compute_cost_test(cost_test_multi_obj)


def test_sigmoidal(x_train, w_train):
    print("------------test_sigmoidal-----------")
    rl = LogisticRegMulti(x_train, w_train, 0, 0, 0)
    rl.sigmoidal(w_train)
    x = 0
    print ('Solución sigmoidal de % da %.1f' % (x, rl.sigmoidal(x)))

def run_gradient_descent(x_train, y_train,alpha = 0.01,iterations=1500,lambda_=0):
    # initialize fitting parameters. Recall that the shape of w is (n,)
    initial_w = np.zeros(x_train.shape[1])
    initial_b = 0.

    print("---- Gradient descent POO--- lamb ",lambda_)
    lr = LogisticRegMulti(x_train,y_train,initial_w,initial_b,lambda_)
    w,b,h,w_init,b_init = lr.gradient_descent(alpha,iterations)
    
    return w, b



def test_gradient_descent(x_train, y_train):
    #print("----- Original Gradient descent------")
    w1, b1 = run_gradient_descent(x_train, y_train,0.01,1500,0)
    print("w,b found by gradient descent with labmda 0 ([ 0.93305656  0.18903186 -0.12087087] 0.4690649858291144):", w1, b1)

    w2, b2 = run_gradient_descent(x_train, y_train,0.01,1500,1)
    print("w,b found by gradient descent with labmda 1 ([ 0.93278101  0.18900175 -0.12080013] 0.46905464164492655):", w2, b2)


# w_train son los datos según el ejercicio opcional
x_train, y_train, w_train = load_data_csv_multi_logistic("./Practica2/Enunciado/data/games-data.csv","score","critics","users","user score")
x_train, mu, sigma = zscore_normalize_features(x_train)
test_cost(x_train, y_train)
test_gradient(x_train, y_train)
test_gradient_descent(x_train, y_train)

# Ejercicio opcional
print(">>>>>>>>>>>>>>>>opcional<<<<<<<<<<<<<<<<<")
clases = []
num_clases = 5
test_w = np.array([1, 0.5, -0.35])
b = 1

# Regresión logistica de cada clase
for k in range(num_clases):
    # Si es el número que estamos buscando lo pone a 0 y si no a 1
    w_k = (w_train == k).astype(int) # Hacemos el vector binario para cada clase
    clase_k = LogisticRegMulti(x_train, w_k, test_w, b, 0)
    # hacemos el gradient descent (Entrenamos los modelos)
    clase_k.gradient_descent(alpha = 0.01,num_iters=1500)
    clases.append(clase_k)

# Probamos el resultado
probs = []
for clase in clases:
    prob = clase.f_w_b(x_train[0]) #Le pedimos que prediga la puntuación de un juego concreto
    probs.append(prob)

prediccion = np.argmax(probs) #Nos da la clase que tiene la probabilidad más alta de ser

#Predice regular pero diría que es porque está muy desvalanceado tanto los números que entran en cada nota como los resultados de las notas de los juegos
print("Predicción:", prediccion, " | Clase real:", w_train)


# Historial de rutas relativas que funcionan
#./Practica2/Enunciado/data/games-data.csv
#./data/games-data.csv