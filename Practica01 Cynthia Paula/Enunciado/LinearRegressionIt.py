import numpy as np
import copy
import math
from utils import *

    
class LinearRegIt:

    """
    Linear regresion en su versión iterativa
    """

    """
    Computes the cost function for linear regression.

    Args:
        x (ndarray): Shape (m,) Input to the model
        y (ndarray): Shape (m,) the real values of the prediction
        w, b (scalar): Parameters of the model
    """
    def __init__(self, x, y,w,b):
        self.x = x      #   Datos de referencia (Los datos reales) 'score'
        self.y = y      #   Datos que queremos predecir 'user Score'
        self.w = w      #   Pendiente de la recta
        self.b = b      #   Bias de la función, necesario para que el error sea mínimo
        

    """
    Computes the linear regression function.

    Args:
        x (ndarray): Shape (m,) Input to the model
    
    Returns:
        the linear regression value
    """
    # Calcula la y_prima del modelo (Los datos que estamos adivinando)
    def f_w_b(self, x):
        
        mul = x.copy()
        for i in range(len(x)):
            mul[i] = mul[i] * self.w

        return mul + self.b


    """
    Computes the cost function for linear regression.

    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    # Analiza como de bien se ajusta el modelo a los datos del entrenamiento,
    # representa la desviación del modelo con respecto a los datos reales
    def compute_cost(self):

        y_prima = self.f_w_b(self.x)

        m = self.x.shape[0]
        sumatorio = 0
        for i in range(m):
            sumatorio += (self.y[i] - y_prima[i]) ** 2
        
        cost = sumatorio / (m * 2)
        return  cost
    

    """
    Computes the gradient for linear regression 
    Args:

    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
     """
    # El gradiente es la derivada parcial de los datos
    # Se utiliza en el descenso de gradiente para ver la derivada de la función en un punto
    def compute_gradient(self):

        y_prima = self.f_w_b(self.x)
        m = self.x.shape[0]
        
        sumw = 0
        for i in range(m):
            sumw += (y_prima[i] - self.y[i]) * self.x[i]
        dj_dw = sumw / m

        sumb = 0
        for i in range(m):
            sumb += (y_prima[i] - self.y[i])
        dj_db = sumb / m

        return dj_dw, dj_db
    
    
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha

    Args:
      alpha : (float) Learning rate
      num_iters : (int) number of iterations to run gradient descent
    Returns
      w : (ndarray): Shape (1,) Updated values of parameters of the model after
          running gradient descent
      b : (scalar) Updated value of parameter of the model after
          running gradient descent
      J_history : (ndarray): Shape (num_iters,) J at each iteration,
          primarily for graphing later
      w_initial : (ndarray): Shape (1,) initial w value before running gradient descent
      b_initial : (scalar) initial b value before running gradient descent
    """
    # Trata de minimizar J(w,b), cambia w y b hasta conseguir el valor más cercano al mínimo posible
    # Para ello se utiliza la función de coste
    # Este algoritmo es el que va viendo la derivada para ver si se ha quedado corto para llegar al objetivo
    # (Derivada decreciente) o se ha pasado (Derivada creciente) acercandose cada vez más al vértice de la función
    # Nuestra respuesta correcta
    def gradient_descent(self, alpha, num_iters):
        # An array to store cost J and w's at each iteration — primarily for graphing later
        J_history = []
        w_history = []
        w_initial = copy.deepcopy(self.w)  # avoid modifying global w within function
        b_initial = copy.deepcopy(self.b)  # avoid modifying global w within function

        for i in range(num_iters):
            w,b = self.compute_gradient()

            self.w = self.w - alpha * w
            self.b = self.b - alpha * b

            J_history.append(self.compute_cost())
        
        return self.w, self.b, J_history, w_initial, b_initial


def cost_test_obj(x,y,w_init,b_init):
    lr = LinearRegIt(x,y,w_init,b_init)
    cost = lr.compute_cost()
    return cost

def compute_gradient_obj(x,y,w_init,b_init):
    lr = LinearRegIt(x,y,w_init,b_init)
    dw,db = lr.compute_gradient()
    return dw,db