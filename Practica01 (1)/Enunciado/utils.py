import numpy as np
import pandas as pd
from csv import reader


# read the CSV file
def load_csv(filename):
    # Open file in read mode
    file = open(filename,"r")
    # Reading file 
    lines = reader(file)
    
    # Converting into a list 
    data = list(lines)

    return data

def cleanData(data):
    data["score"] = data["score"].apply(lambda x:  str(x).replace(",","."))
    data = data.drop(data[data["user score"] == "tbd"].index)
    data["user score"] = data["user score"].apply(lambda x:  str(x).replace(",","."))
    data["score"] = data["score"].astype(np.float64) / 10
    data["user score"] = data["user score"].astype(np.float64)

    return data

def cleanDataMulti(data):
    data = cleanData(data)
    data["critics"] = data["critics"].astype(np.float64)
    data["users"] = data["users"].astype(np.float64)
    return data

def load_data_csv(path,x_colum,y_colum):
    data = pd.read_csv(path)
    data = cleanData(data)
    X = data[x_colum].to_numpy()
    y = data[y_colum].to_numpy()

    #Poner dentro de Clean data 
    #y_ = y * 10
    return X, y


# Cynthia dice: holaaa he visto que esto está en
# la diapo 13 del tema 2 reg multivariable y en wikipedia (standard score)
def zscore_normalize_features(X):
    """
    computes  X, zcore normalized by column

    Args:
      X (ndarray (m,n))     : input data, m examples, n features

    Returns:
      X_norm (ndarray (m,n)): input normalized by column
      mu (ndarray (n,))     : mean of each feature
      sigma (ndarray (n,))  : standard deviation of each feature
    """
    
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
    data = load_csv(path)
    #data = pd.read_csv(path)
    data = cleanDataMulti(data)
    x1 = data[x1_colum].to_numpy()
    x2 = data[x2_colum].to_numpy()
    x3 = data[x3_colum].to_numpy()
    X = np.array([x1, x2, x3])
    X = X.T
    y = data[y_colum].to_numpy()
    return X, y

def GetNumGradientsSuccess(w1,w1Sol,b1,b1Sol):
    iterator = 0
    for i in range(len(w1)): 
        if np.isclose(w1[i],w1Sol[i]):
                iterator += 1
    if np.isclose(b1,b1Sol):
        iterator += 1
    return iterator