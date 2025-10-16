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
        
        #cost = -1/m * np.sum(self.y * np.log(y_prima) + (1 - self.y * np.log(1 - y_prima))) #lo que leo en el enunciado
        cost = self.y * np.log(y_prima) + (1 - self.y) * np.log(1 - y_prima) # lo de la url y la correcta
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
