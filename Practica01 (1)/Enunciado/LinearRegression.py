import numpy as np
import copy
import math
from utils import *

class LinearReg:
    """
    Computes the cost function for linear regression.

    Args:
        x (ndarray): Shape (m,) Input to the model
        y (ndarray): Shape (m,) the real values of the prediction
        w, b (scalar): Parameters of the model
    """
    def __init__(self, x, y,w,b):
        #(scalar): Parameters of the model
        #d = [w,b]
        #da = np.array(d)
        self.x = x  #Datos de referencia (Los datos reales) 'score'
        self.y = y  #Datos que queremos predecir 'user Score'
        self.w = w
        self.b = b
        

    """
    Computes the linear regression function.

    Args:
        x (ndarray): Shape (m,) Input to the model
    
    Returns:
        the linear regression value
    """
    def f_w_b(self, x):
        
        mul = np.multiply(self.w, x)
        return mul + self.b


    """
    Computes the cost function for linear regression.

    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    def compute_cost(self):

        y_prima = self.f_w_b(self.x)
        cost = np.sum(np.square(self.y - y_prima))/(np.size(self.y) * 2)
        #                   m                             y      y'
        #return (1 / 2 * np.size(self.y)) * np.sum(np.square(self.y - y_prima))
        return  cost
    

    """
    Computes the gradient for linear regression 
    Args:

    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
     """
    def compute_gradient(self):

        y_prima = self.f_w_b(self.x)
                #           m
        dj_dw =  ((1/ np.size(self.y)) * (np.sum((y_prima - self.y) * self.x)))
        dj_db = ((1/ np.size(self.y)) * (np.sum(y_prima - self.y)))
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
    lr = LinearReg(x,y,w_init,b_init)
    cost = lr.compute_cost()
    return cost

def compute_gradient_obj(x,y,w_init,b_init):
    lr = LinearReg(x,y,w_init,b_init)
    dw,db = lr.compute_gradient()
    return dw,db