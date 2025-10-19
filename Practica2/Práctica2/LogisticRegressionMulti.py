import numpy as np
import copy
import math

from LinearRegressionMulti import LinearRegMulti

class LogisticRegMulti(LinearRegMulti):

    """
    Computes the cost function for linear regression.

    Args:
        x (ndarray): Shape (m,) Input to the model
        y (ndarray): Shape (m,) the real values of the prediction
        w, b (scalar): Parameters of the model
        lambda: Regularization parameter. Most be between 0..1. 
        Determinate the weight of the regularization.
    """
    def __init__(self, x, y,w,b, lambda_):
        super().__init__(x, y,w,b,lambda_)


    """
    Computes the linear regression function.

    Args:
        x (ndarray): Shape (m,) Input to the model
    
    Returns:
        the linear regression value
    """
    def f_w_b(self, x):
        ex = super().f_w_b(x)
        return 1/(1 + (np.exp(-ex)))
 
# En el enunciado leo la ecuacion de una manera distinta a la que pone aquí:   
#https://www.geeksforgeeks.org/machine-learning/ml-cost-function-in-logistic-regression/
    def compute_cost(self):
        y_prima = self.f_w_b(self.x) # y predicha
        m = self.m
        y = self.y
        
        # PROBLEMA!!! 
        # posibilidad de logaritmo  de 0 o negativo
        # lo importante de la función de coste es:
        # y * np.log(y_prima) + (1 - y * np.log(1 - y_prima))
        # en la versión logística binaria hacíamos:
        # si y = 0 e y' = 0 -> 0 * log(0) + 1 * log(1)
        # si y = 0 e y' = 1 -> 0 * log(1) + 1 * log(0) # esto debería haber dado error de por si en la versión binaria... creo
        # si y = 1 e y' = 0 -> 1 * log(0) + 0 * log(1)
        # si y = 1 e y' = 1 -> 1 * log(1) + 0 * log(0)
        # imagino que numpy hace algún tipo de omisión a log(0) pero con las variables no binarias está dando logaritmo negativo
        
        # lo único que se me ocurre es clampear
        # al parecer los float64 de numpy tienen una precisión de -15 a +16
        # https://www.pythoninformer.com/python-libraries/numpy/data-types/#floats
        # si uso exponente 16 funciona pero voy a usar 15 por si acaso XD 
        #clip es el equivalente a clamp en numpy
        #clamp = 1e-15
        #y_prima = np.clip(y_prima, clamp, 1 - clamp) #es o 0.0x o 0.9x
        
        #cost = -1/m * np.sum(y * np.log(y_prima) + (1 - y * np.log(1 - y_prima))) #lo que leo en el enunciado
        cost = y * np.log(y_prima) + (1 - y) * np.log(1 - y_prima) # lo de la url y la correcta
        loss = np.sum(cost)
        total_loss = (-1 / m * loss)
        
        total_loss += self._regularizationL2Cost()
        return  total_loss
    
    def sigmoidal(self):
        sig = 1 / (1 + np.exp(-self.x))
        return sig
    
def cost_test_multi_obj(x,y,w_init,b_init):
    lr = LogisticRegMulti(x,y,w_init,b_init,0)
    cost = lr.compute_cost()
    return cost

def compute_gradient_multi_obj(x,y,w_init,b_init):
    lr = LogisticRegMulti(x,y,w_init,b_init,0)
    dw,db = lr.compute_gradient()
    return dw,db
