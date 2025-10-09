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

    def f_w_b(self, x):
        ex = super().f_w_b(x)
        return 1/(1 + (np.exp(-ex)));

    def compute_cost(self):
        y_prima = self.f_w_b(self.x) # y predicha
        # m = self.m
        # m, c = np.polyfit(self.x, self.y, 1)
        # m = np.gradient(self.x);
        m = self.m
        cost = (1/2*m) * np.sum((y_prima - self.y)**2)
        # esto vendrá después para el ej 3
        cost += self._regularizationL2Cost()
        return  cost
 
# En el enunciado leo la ecuacion de una manera distinta a la que pone aquí:   
#https://www.geeksforgeeks.org/machine-learning/ml-cost-function-in-logistic-regression/
    def compute_cost(self):
        y_prima = self.f_w_b(self.x) # y predicha
        m = self.m
        
        #cost = -1/m * np.sum(self.y * np.log(y_prima) + (1 - self.y * np.log(1 - y_prima))) #lo que leo en el enunciado
        cost = -1/m * np.sum(self.y * np.log(y_prima) + (1 - self.y) * np.log(1 - y_prima)) # lo de la url y la correcta
        
        #cost += self._regularizationL2Cost()
        return  cost
    
    ### Creo que no hace falta overridear ni esta función ni la del descenso ni la de L2, ya pasa los 3 tests con la herencia...
    # def compute_gradient(self): 
    #     y_prima = self.f_w_b(self.x)
    #     error = y_prima - self.y
    #     m = self.m

    #     dj_dw = (1/m) * (self.x.T @ error)   # la traspuesta por el error
    #     dj_db = (1/m) * np.sum(error)  #     

    #     # esto vendrá después para el ej 3
    #     dj_dw  += self._regularizationL2Gradient()
        
    #     return dj_dw, dj_db
    
def cost_test_multi_obj(x,y,w_init,b_init):
    lr = LogisticRegMulti(x,y,w_init,b_init,0)
    cost = lr.compute_cost()
    return cost

def compute_gradient_multi_obj(x,y,w_init,b_init):
    lr = LogisticRegMulti(x,y,w_init,b_init,0)
    dw,db = lr.compute_gradient()
    return dw,db
