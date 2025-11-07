from MLP import MLP, target_gradient, costNN, MLP_backprop_predict
from utils import load_data, load_weights,one_hot_encoding, accuracy
from public_test import checkNNGradients,MLP_test_step
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import random
import numpy as np



"""
Test 1 to be executed in Main
"""
def gradientTest():
    checkNNGradients(costNN,target_gradient,0)
    checkNNGradients(costNN,target_gradient,1)


"""
Test 2 to be executed in Main
"""
def MLP_test(X_train,y_train, X_test, y_test):
    print("We assume that: random_state of train_test_split  = 0 alpha=1, num_iterations = 2000, test_size=0.33, seed=0 and epislom = 0.12 ")
    print("Test 1 Calculando para lambda = 0")
    MLP_test_step(MLP_backprop_predict,1,X_train,y_train,X_test,y_test,0,2000,0.92606,2000/10)
    print("Test 2 Calculando para lambda = 0.5")
    MLP_test_step(MLP_backprop_predict,1,X_train,y_train,X_test,y_test,0.5,2000,0.92545,2000/10)
    print("Test 3 Calculando para lambda = 1")
    MLP_test_step(MLP_backprop_predict,1,X_train,y_train,X_test,y_test,1,2000,0.92667,2000/10)
    
def SKLearn_test(X_train, Y_train, X_test, Y_test):
    n_hidden_neurons = 25 # numero de neuronas de la capa oculta
    lambda_ = 0.0
    alpha = 1.0
    num_ite = 2000 
    mlp_sklearn = MLPClassifier(
        hidden_layer_sizes = (n_hidden_neurons,),
        activation = 'logistic',   # sigmoidal
        # solver='adam',
        solver = 'sgd', # en vez de adam=???
        alpha = lambda_,           # regularización L2 alfa como lambda
        learning_rate_init = alpha,
        max_iter = num_ite, # numero de iteraciones
        random_state = 42, # aleatoriedad
        # tol=0.0, # tolerancia minima (diferencia mínima entre el coste de la iteración anterior para que la optimización se detenga si no la supera tras 10 iteraciones)
        verbose = False # si escribe por consola mensajes de debug
    )
    mlp_sklearn.fit(X_train, Y_train)
    Y_pred_sklearn = mlp_sklearn.predict(X_test)
    acc_sklearn = accuracy_score(Y_test, Y_pred_sklearn)
    print(f"SKLEARN: Calculated accuracy for lambda = {(lambda_):1.5f} : {(acc_sklearn):1.5f}")
    
    lambda_ = 0.5
    mlp_sklearn = MLPClassifier(
        hidden_layer_sizes = (n_hidden_neurons,),
        activation = 'logistic',  
        # solver='adam',
        solver = 'sgd', 
        alpha = lambda_,           
        learning_rate_init = alpha,
        max_iter = num_ite, 
        random_state = 42, 
        # tol=0.0, 
        verbose = False 
    )
    mlp_sklearn.fit(X_train, Y_train)
    Y_pred_sklearn = mlp_sklearn.predict(X_test)
    acc_sklearn = accuracy_score(Y_test, Y_pred_sklearn)
    print(f"SKLEARN: Calculated accuracy for lambda = {(lambda_):1.5f} : {(acc_sklearn):1.5f}")
    
    lambda_ = 1.0
    mlp_sklearn = MLPClassifier(
        hidden_layer_sizes = (n_hidden_neurons,),
        activation = 'logistic',  
        # solver='adam',
        solver = 'sgd',
        alpha = lambda_,           
        learning_rate_init = alpha,
        max_iter = num_ite, 
        random_state = 42, 
        # tol=0.0,
        verbose = False
    )
    mlp_sklearn.fit(X_train, Y_train)
    Y_pred_sklearn = mlp_sklearn.predict(X_test)
    acc_sklearn = accuracy_score(Y_test, Y_pred_sklearn)
    print(f"SKLEARN: Calculated accuracy for lambda = {(lambda_):1.5f} : {(acc_sklearn):1.5f}")

def Our_test(X_train, y_train_encoded, X_test, Y_test):
    lambda_ = 0.0
    alpha = 1.0
    num_ite = 2000 
    lambda_ = 0.0
    y_pred = MLP_backprop_predict(X_train, y_train_encoded, X_test, alpha, lambda_, num_ite, 0)
    accu = accuracy_score(Y_test, y_pred)
    print(f"OURS: Calculated accuracy for lambda = {(lambda_):1.5f} : {(accu):1.5f}")
    
    lambda_ = 0.5
    y_pred = MLP_backprop_predict(X_train, y_train_encoded, X_test, alpha, lambda_, num_ite, 0)
    accu = accuracy_score(Y_test, y_pred)
    print(f"OURS: Calculated accuracy for lambda = {(lambda_):1.5f} : {(accu):1.5f}")
    
    lambda_ = 1.0
    y_pred = MLP_backprop_predict(X_train, y_train_encoded, X_test, alpha, lambda_, num_ite, 0)
    accu = accuracy_score(Y_test, y_pred)
    print(f"OURS: Calculated accuracy for lambda = {(lambda_):1.5f} : {(accu):1.5f}")


def main():
    print("Main program")

    # 400 es el tamaño del input (imagenes de 20 x 20 pixeles)
    # 25 es el tamaño de la capa oculta
    # 10 es el tamaño del output (Los 10 dígitos [0 - 9])
    appleJack = MLP(400, 25, 10)
    
    #Test 1
    gradientTest()

    # Cargamos los datos reales
    X, Y = load_data('./data/ex3data1.mat')

    # Hay que coger una parte aleatoria de los datos, preferiblemente aleatorio a una sección para evitar sesgos
    # Cogemos una muestra aleatoria de los datos para entrenamiento y para los test
    X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, shuffle=True, random_state=42
)
    # Pasamos SOLO LOS DATOS DE ENTRENAMIENTO por el one_hot_encoding (la salida) para que la codificación coincida
    y_train_encoded = one_hot_encoding(Y_train)
    
    #Test 2
    # Pasamos el test
    #MLP_test(X_train, y_train_encoded, X_test, Y_test)







    # Ejercicio 4: MLP de sklearn
    # https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
    
    # los test de MLP pasan con 3 lambas distintas (0, 0.5, 1) según estos parámetros, para lambda = 1:
    # MLP_test_step(MLP_backprop_predict,1,X_train,y_train,X_test,y_test, lambda,2000,0.92667,2000/10)
    # alfa = 1, 
    # num_ite = 2000
    # baseLineAccuracy = 0.92667
    # verbose = 2000/10
    
    SKLearn_test(X_train, Y_train, X_test, Y_test)
    
    # nuestra precisión
    Our_test(X_train, y_train_encoded, X_test, Y_test)
main()