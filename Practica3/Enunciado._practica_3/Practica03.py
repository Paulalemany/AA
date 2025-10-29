import numpy as np
import matplotlib.pyplot as plt
from utils import load_data, load_weights,one_hot_encoding, accuracy, confMatrix, F1Score
from MLP import MLP
from public_test import compute_cost_test, predict_test


x,y = load_data('Enunciado._practica_3/data/ex3data1.mat')
theta1,theta2 = load_weights('Enunciado._practica_3/data/ex3weights.mat')

#Ejercicio 1
Pinkypie = MLP(theta1, theta2)
a1,a2,a3,z2,z3 = Pinkypie.feedforward(x)
p = Pinkypie.predict(a3)
predict_test(p, y, accuracy)

#Ejercicio 2
y_o_h = one_hot_encoding(y)
compute_cost_test(Pinkypie, a3, y_o_h)

#Ejercicio 3
#Tenemos que hacer la matriz para el 0
y_binario = (y == 0).astype(int)    # Si el valor es 0 es correcto
p_binario = (p == 0).astype(int)  

cm = confMatrix(y_binario, p_binario)
tn, fp, fn, tp = cm.ravel().tolist()
f1 = F1Score(tp, fp, fn)
print(cm)
print (f1)