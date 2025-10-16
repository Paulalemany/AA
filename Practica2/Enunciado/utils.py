import numpy as np
import pandas as pd


def cleanData(data):
    data["score"] = data["score"].apply(lambda x:  str(x).replace(",","."))
    data = data.drop(data[data["user score"] == "tbd"].index)
    data["user score"] = data["user score"].apply(lambda x:  str(x).replace(",","."))
    data["score"] = data["score"].astype(np.float64)
    data["user score"] = data["user score"].astype(np.float64)
    data["critics"] = data["critics"].astype(np.float64)
    data["users"] = data["users"].astype(np.float64)
    return data

def load_data_csv(path,x_colum,y_colum):
    data = pd.read_csv(path)
    data = cleanData(data)
    X = data[x_colum].to_numpy()
    y = data[y_colum].to_numpy()
    return X, y

def zscore_normalize_features(X):
    # find the mean of each column/feature
    # mu will have shape (n,)
    mu = np.mean(X, axis=0)
    # find the standard deviation of each column/feature
    # sigma will have shape (n,)
    sigma = np.std(X, axis=0, ddof=0) 
    # element-wise, subtract mu for that column from each example,
    # divide by std for that column (cyntrist: <-- sigma entiendo (standard deviation))

    X_norm = (X - mu) / sigma

    return X_norm, mu, sigma

def load_data_csv_multi(path,x1_colum,x2_colum,x3_colum,y_colum):
    data = pd.read_csv(path)
    data = cleanData(data)
    x1 = data[x1_colum].to_numpy()
    x2 = data[x2_colum].to_numpy()
    x3 = data[x3_colum].to_numpy()
    X = np.array([x1, x2, x3])
    X = X.T
    y = data[y_colum].to_numpy()
    return X, y

    
## 0 Malo, 1 Regular, 2 Notable, 3 Sobresaliente, 4 Must Play.
def load_data_csv_multi_logistic(path,x1_colum,x2_colum,x3_colum,y_colum):
    X, y = load_data_csv_multi(path,x1_colum,x2_colum,x3_colum,y_colum)

    w = y
    m = y.shape[0]
    # funcion chula es como el operador ternario pero para un array entero 
    # (crea un array del mismo tamaño que y, si y[i] < 7 lo rellena con 0 y si no con 1)
    y = np.where(y < 7.0, 0, 1) 

    # Para el ejercicio opcional
    for i in range(m):
        if w[i] < 5:
            w[i] = 0
        elif w[i] < 7:
            w[i] = 1
        elif w[i] < 9:
            w[i] = 2
        elif w[i] < 9.5:
            w[i] = 3
        else:
            w[i] = 4


    return X, y, w
        
    
        