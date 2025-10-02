import numpy as np
import copy
import math
from utils import * 

class LinearRegNumpy:
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
        return self.w * x + self.b


    """
    Computes the cost function for linear regression.

    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    def compute_cost(self):

        y = self.f_w_b(self.x)
        #                   m                             y      y'
        return (1 / 2 * len(self.x)) * np.sum(np.square(self.x - y))
    

    """
    Computes the gradient for linear regression 
    Args:

    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
     """
    def compute_gradient(self):

        dj_dw = np.gradient(self.w)
        dj_db = np.gradient(self.b)
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
        #TODO: gradient descent iteration by m examples.

        #Aunque no estemos haciendo las cosas de forma iterativa las Epocs deben hacerse iterativamente
        #No estoy segura de si debe ser self.w = w_initial o deben ser w_initial = w_initial
        for i in num_iters:
            self.w = w_initial - alpha * (self.compute_gradient().index(0))
            self.b = b_initial - alpha * (self.compute_gradient().index(1))

            #J_history = self.compute_cost()
            J_history.append(self.compute_cost())
        
        return self.w, self.b, J_history, w_initial, b_initial


def cost_test_obj(x,y,w_init,b_init):
    lr = LinearRegNumpy(x,y,w_init,b_init)
    cost = lr.compute_cost()
    return cost

def compute_gradient_obj(x,y,w_init,b_init):
    lr = LinearRegNumpy(x,y,w_init,b_init)
    dw,db = lr.compute_gradient()
    return dw,db
