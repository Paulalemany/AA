from MLP import MLP, target_gradient, costNN, MLP_backprop_predict
from utils import load_data, load_weights,one_hot_encoding, accuracy
from public_test import checkNNGradients,MLP_test_step
from sklearn.model_selection import train_test_split
import random



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



def main():
    print("Main program")

    # 400 es el tamaño del input (imagenes de 20 x 20 pixeles)
    # 25 es el tamaño de la capa oculta
    # 10 es el tamaño del output (Los 10 dígitos [0 - 9])
    appleJack = MLP(400, 25, 10)
    
    #Test 1
    gradientTest()

    # Cargamos los datos reales
    X, Y = load_data('Practica04Enunciado/Practica04Enunciado/data/ex3data1.mat')

    # Hay que coger una parte aleatoria de los datos, preferiblemente aleatorio a una sección para evitar sesgos
    # Cogemos una muestra aleatoria de los datos para entrenamiento y para los test
    X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, shuffle=True, random_state=42
)
    # Pasamos SOLO LOS DATOS DE ENTRENAMIENTO por el one_hot_encoding (la salida) para que la codificación coincida
    y_train_encoded = one_hot_encoding(Y_train)
    
    #Test 2
    # Pasamos el test
    MLP_test(X_train, y_train_encoded, X_test, Y_test)

    

main()