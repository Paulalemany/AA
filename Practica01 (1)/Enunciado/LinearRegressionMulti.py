import numpy as np
import copy
import math

from LinearRegression import LinearReg

class LinearRegMulti(LinearReg):

    """
    Computes the cost function for linear regression.

    Args:
        x (ndarray): Shape (m,) Input to the model
        y (ndarray): Shape (m,) the real values of the prediction
        w, b (scalar): Parameters of the model
        lambda: Regularization parameter. Most be between 0..1. 
        Determinate the weight of the regularization.
    """
    def __init__(self, x, y, w, b, lambda_):
        super().__init__(x, y, w, b)
        self.x = x
        self.y = y
        self.w = w
        self.b = b
        self.lambda_ = lambda_
        return

    def f_w_b(self, x):
        ret = x @ self.w + self.b
        return ret

    
    """
    Compute the regularization cost (is private method: start with _ )
    This method will be reuse in the future.

    Returns
        _regularizationL2Cost (float): the regularization value of the current model
    """
    def _regularizationL2Cost(self):
        reg_cost_final = 0
        return reg_cost_final
    
    """
    Compute the regularization gradient (is private method: start with _ )
    This method will be reuse in the future.

    Returns
        _regularizationL2Gradient (vector size n): the regularization gradient of the current model
    """ 
    def _regularizationL2Gradient(self):
        reg_gradient_final = 0
        return reg_gradient_final


    ################# Herencia
    """
    Computes the cost function for linear regression.

    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    # https://medium.com/data-science/applied-multivariate-regression-faef8ddbf807
    def compute_cost(self):
        y_prima = self.f_w_b(self.x) # y predicha
        m = self.m
        cost = (1/2*m) * np.sum((y_prima - self.y)**2)
        if self.lambda_ > 0: # esto vendrá después para el ej 3
            cost += self._regularizationL2Cost()
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
        error = y_prima - self.y
        
        #gradiente w y b
        # dj_dw =  ((1/ np.size(self.y)) * (np.sum((error) * self.x)))
        # dj_db = ((1/ np.size(self.y)) * (np.sum(error)))

        if self.lambda_ > 0: # esto vendrá después para el ej 3
            dj_dw  += self._regularizationL2Gradient()
        
        return 0, 0 
    
    
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
    # https://medium.com/@IwriteDSblog/gradient-descent-for-multivariable-regression-in-python-d430eb5d2cd8
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
    #################
    
def cost_test_multi_obj(x,y,w_init,b_init):
    lr = LinearRegMulti(x,y,w_init,b_init,0)
    cost = lr.compute_cost()
    return cost

def compute_gradient_multi_obj(x,y,w_init,b_init):
    lr = LinearRegMulti(x,y,w_init,b_init,0)
    dw,db = lr.compute_gradient()
    return dw,db
