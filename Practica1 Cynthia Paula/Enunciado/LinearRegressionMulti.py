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
        self.m = self.x.shape[0]
        self.lambda_ = lambda_   #   Determina el peso de cada argumento de la función
        return


    """
    Computes the linear regression function.

    Args:
        x (ndarray): Shape (m,) Input to the model
    
    Returns:
        the linear regression value
    """
    # Cálculo de la y_prima del modelo (De forma matricial)
    # Luego se compaparán con los datos reales (y) para ver como de lejos ha estado el modelo de la realidad
    def f_w_b(self, x):
        ret = x @ self.w + self.b
        return ret

    
    """
    Compute the regularization cost (is private method: start with _ )
    This method will be reuse in the future.

    Returns
        _regularizationL2Cost (float): the regularization value of the current model
    """
    #https://developers.google.com/machine-learning/crash-course/overfitting/regularization?hl=es-419
    def _regularizationL2Cost(self):
        reg_cost_final = (self.lambda_ / 2 * self.m) * np.sum(self.w**2)
        return reg_cost_final
    
    """
    Compute the regularization gradient (is private method: start with _ )
    This method will be reuse in the future.

    Returns
        _regularizationL2Gradient (vector size n): the regularization gradient of the current model
    """ 
    def _regularizationL2Gradient(self):
        reg_gradient_final = self.lambda_ / self.m * self.w 
        return reg_gradient_final


    """
    Computes the cost function for linear regression.

    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    # https://medium.com/data-science/applied-multivariate-regression-faef8ddbf807
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
    
    """
    Computes the gradient for linear regression 
    Args:

    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
     """
    # El gradiente es equivalente a la derivada de una variable, aplicada múltiples variables. Se resuelve produciendo un vector con tantas dimensiones como variables haya. En cada dimensión se calculará la derivada parcial con respecto a una de las variables. Indica la dirección de máximo cambio de la función. La magnitud del gradiente es la pendiente de la gráfica en esa dirección. El gradiente se representa con el operador diferencial nabla
    def compute_gradient(self):
        y_prima = self.f_w_b(self.x)
        error = y_prima - self.y
        m = self.m

        dj_dw = (1/m) * (self.x.T @ error)   # la traspuesta por el error
        dj_db = (1/m) * np.sum(error)  #     

        # esto vendrá después para el ej 3
        dj_dw  += self._regularizationL2Gradient()
        
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

            cost = self.compute_cost()
            J_history.append(cost)

        return self.w, self.b, J_history, w_initial, b_initial

    
def cost_test_multi_obj(x,y,w_init,b_init):
    lr = LinearRegMulti(x,y,w_init,b_init,0)
    cost = lr.compute_cost()
    return cost

def compute_gradient_multi_obj(x,y,w_init,b_init):
    lr = LinearRegMulti(x,y,w_init,b_init,0)
    dw,db = lr.compute_gradient()
    return dw,db
