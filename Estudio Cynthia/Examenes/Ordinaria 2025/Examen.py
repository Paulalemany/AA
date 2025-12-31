from MLP_Complete import MLP_Complete
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from Utils import one_hot_encoding

#LECTURA
df = pd.read_csv("./Examenes/Ordinaria 2025/data/heart.csv")                      

#region EJERCICIO 1: LIMPIEZA Y PREPROCESADO
df = df.dropna(how="any") # deshacernos de NaN
clase = "HeartDisease" # atributo objetivo

ohe_columns = [ # variables categóricas de mas de dos tipos
    "ChestPainType",
    "RestingECG",
    "ST_Slope"
    ]
label_columns = [ # variables categoricas binarias
    "Sex",
    "ExerciseAngina"
]
sc_columns = df.drop(columns=ohe_columns + label_columns + [clase]).columns # las demas, escalares
#otra manera de hacerlo:
# sc_columns = [
#     col for col in df.columns
#     if col not in ohe_columns + label_columns + [clase]
# ]
# print("OHE:", ohe_columns)
# print("Label:", label_columns)
# print("Scaled:", sc_columns.tolist())

# OHE:
ohe = OneHotEncoder(sparse_output=False)
ohe_data = ohe.fit_transform(df[ohe_columns])
ohe_feature_names = ohe.get_feature_names_out(ohe_columns)
df_ohe = pd.DataFrame(ohe_data, columns=ohe_feature_names, index=df.index)

# LABEL ENCODER
le = LabelEncoder() 
for col in label_columns:
    df[col] = le.fit_transform(df[col])
#otra manera de hacerlo:
# binary_mapping = {
#     "Sex": {"M": 1, "F": 0},
#     "ExerciseAngina": {"Y": 1, "N": 0}
# }
# for col in label_columns:
#     df[col] = df[col].map(binary_mapping[col])

    
# STANDARD SCALER
scaler = StandardScaler()
sc_data = scaler.fit_transform(df[sc_columns])
df_sc = pd.DataFrame(sc_data, columns=sc_columns, index=df.index)


X = pd.concat([df_ohe, df_sc, df[label_columns]], axis=1) 
y = le.fit_transform(df[clase])
#endregion



#region EJERCICIO 2: DIBUJADO
# x = df.drop(columns=[clase]) 
# scaling = StandardScaler()        
# x_scaled = scaling.fit_transform(X)    
pca = PCA(n_components=2) 
x_pca = pca.fit_transform(X)
df_pca = pd.DataFrame({
    "PC1": x_pca[:, 0],
    "PC2": x_pca[:, 1],
    clase: y
})

grupos = sorted(df_pca[clase].unique())
colormap = matplotlib.colormaps.get_cmap("prism")
colors = colormap(np.linspace(1, 0, len(grupos)))

fig = plt.figure(figsize=(8,6)) 
for idx, grupo in enumerate(grupos):
    subset = df_pca[df_pca[clase] == grupo]
    plt.scatter(
        subset["PC1"], 
        subset["PC2"],
        label=f"Grupo {grupo}",
        color = colors[idx],
        alpha = 0.7
        )
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("PCA Heart Disease")
plt.legend()
#plt.show()
#endregion


# DATOS!!!
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25)

#region EJERCICIO 3
y_train_encoded = one_hot_encoding(y_train)

x_train_np = X_train.to_numpy()
x_test_np = X_test.to_numpy()

alpha = 0.5
num_ite = 2000 
lambda_ = 0.5
print(f"________MLP LOGISTIC 3 CAPAS________" )
start = time.time()
mlp_complete = MLP_Complete(
    inputLayer=x_train_np.shape[1], 
    hiddenLayers=[128, 64, 32], 
    outputLayer=y_train_encoded.shape[1]
    )
Jhistory = mlp_complete.backpropagation(x_train_np,y_train_encoded,alpha,lambda_,num_ite, verbose=100)
a_list, z_list = mlp_complete.feedforward(x_test_np)
a3 = a_list[-1]   # activación de la última capa
y_pred = mlp_complete.predict(a3)

acc_complete = accuracy_score(y_test, y_pred) #precision¡
print(f"MLP Accuracy for Lambda = {(lambda_):1.5f} : {(acc_complete):1.5f}")
cfm_mlp_complete = confusion_matrix(y_test, y_pred) # la nuestra
print("MLP COMPLETE Confusion Matrix:\n", cfm_mlp_complete)
print(f"________FIN________" )
end = time.time()
print(f"\n\tDuración {(end - start):1.5f} s\n")

print(f"________MLP TANH 3 CAPAS________" )
start = time.time()
mlp_complete = MLP_Complete(
    inputLayer=x_train_np.shape[1], 
    hiddenLayers=[128, 64, 32], 
    outputLayer=y_train_encoded.shape[1],
    hidden_function="tanh"
    )
Jhistory = mlp_complete.backpropagation(x_train_np,y_train_encoded,alpha,lambda_,num_ite, verbose=100)
a_list, z_list = mlp_complete.feedforward(x_test_np)
a3 = a_list[-1]   # activación de la última capa
y_pred = mlp_complete.predict(a3)

acc_complete = accuracy_score(y_test, y_pred) #precision¡
print(f"MLP Accuracy for Lambda = {(lambda_):1.5f} : {(acc_complete):1.5f}")
cfm_mlp_complete = confusion_matrix(y_test, y_pred) # la nuestra
print("MLP COMPLETE Confusion Matrix:\n", cfm_mlp_complete)
print(f"________FIN________" )
end = time.time()
print(f"\n\tDuración {(end - start):1.5f} s\n")

print(f"________MLP RELU 3 CAPAS________" )
start = time.time()
mlp_complete = MLP_Complete(
    inputLayer=x_train_np.shape[1], 
    hiddenLayers=[128, 64, 32], 
    outputLayer=y_train_encoded.shape[1],
    hidden_function="relu"
    )
Jhistory = mlp_complete.backpropagation(x_train_np,y_train_encoded,alpha,lambda_,num_ite, verbose=100)
a_list, z_list = mlp_complete.feedforward(x_test_np)
a3 = a_list[-1]   # activación de la última capa
y_pred = mlp_complete.predict(a3)

acc_complete = accuracy_score(y_test, y_pred) #precision¡
print(f"MLP Accuracy for Lambda = {(lambda_):1.5f} : {(acc_complete):1.5f}")
cfm_mlp_complete = confusion_matrix(y_test, y_pred) # la nuestra
print("MLP COMPLETE Confusion Matrix:\n", cfm_mlp_complete)
print(f"________FIN________" )
end = time.time()
print(f"\n\tDuración {(end - start):1.5f} s\n")

alpha = 0.5
num_ite = 2000 
lambda_ = 0.5
print(f"________MLP 1 CAPA________" )
start = time.time()
mlp_complete = MLP_Complete(
    inputLayer=x_train_np.shape[1], 
    hiddenLayers=[256], 
    outputLayer=y_train_encoded.shape[1]
    )
Jhistory = mlp_complete.backpropagation(x_train_np,y_train_encoded,alpha,lambda_,num_ite, verbose=100)
a_list, z_list = mlp_complete.feedforward(x_test_np)
a3 = a_list[-1]   # activación de la última capa
y_pred = mlp_complete.predict(a3)

acc_complete = accuracy_score(y_test, y_pred) #precision¡
print(f"MLP Accuracy for Lambda = {(lambda_):1.5f} : {(acc_complete):1.5f}")
cfm_mlp_complete = confusion_matrix(y_test, y_pred) # la nuestra
print("MLP COMPLETE Confusion Matrix:\n", cfm_mlp_complete)
print(f"________FIN MLP 1 CAPA________" )
end = time.time()
print(f"\n\tDuración {(end - start):1.5f} s\n")
#endregion

#region EJERCICIO 4
alpha = 0.015
num_ite = 2000 
lambda_ = 0.5
mlp_skl= MLPClassifier(
    hidden_layer_sizes=(256),
    activation='relu',           
    alpha=lambda_,                
    learning_rate_init = alpha,
    max_iter=num_ite,             
    random_state=69,
    )

mlp_skl.fit(X_train, y_train)
y_pred_sklearn = mlp_skl.predict(X_test)
acc_sklearn = accuracy_score(y_test, y_pred_sklearn) #precision
print(f"SKLEARN MLP accuracy for lambda = {(lambda_):1.5f} : {(acc_sklearn):1.5f}")

#endregion

#region EJERCICIO 5
knn = KNeighborsClassifier(
    n_neighbors=7,
    weights='distance',
    p=2,
    metric='minkowski',
    n_jobs=-1
)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_knn)
print(f"KNN Accuracy: {accuracy:.5f}")
#endregion

#region EJERCICIO 6
forest = RandomForestClassifier(
    random_state=420,
    n_jobs=-1              # usar todos los cores disponibles
)
forest.fit(X_train, y_train)
y_pred_forest = forest.predict(X_test)
acc_forest = accuracy_score(y_test, y_pred_forest)
print(f"Random Forest accuracy: {acc_forest:.5f}")
#endregion

#region EJERCICIO 7
cfm_mlp_sk = confusion_matrix(y_test, y_pred_sklearn)
cfm_mlp_complete = confusion_matrix(y_test, y_pred) # la nuestra
cfm_knn = confusion_matrix(y_test, y_pred_knn)
cfm_forest = confusion_matrix(y_test, y_pred_forest)

print("SKLEARN MLP LOGISTIC Confusion Matrix:\n", cfm_mlp_sk)
print("MLP COMPLETE Confusion Matrix:\n", cfm_mlp_complete)
print("KNN Confusion Matrix:\n", cfm_knn)
print("RANDOM FOREST Confusion Matrix:\n", cfm_forest)
#endregion

#region EJERCICIO 8
alpha = 0.5
num_ite = 2000 
lambda_ = 0.5
print(f"________MLP SOFTMAX________" )
start = time.time()
mlp_complete = MLP_Complete(
    inputLayer=x_train_np.shape[1], 
    hiddenLayers=[256], 
    outputLayer=y_train_encoded.shape[1],
    output="softmax"
    )
Jhistory = mlp_complete.backpropagation(x_train_np,y_train_encoded,alpha,lambda_,num_ite, verbose=100)
a_list, z_list = mlp_complete.feedforward(x_test_np)
a3 = a_list[-1]   # activación de la última capa
y_pred = mlp_complete.predict(a3)

acc_complete = accuracy_score(y_test, y_pred) #precision¡
print(f"MLP Accuracy for Lambda = {(lambda_):1.5f} : {(acc_complete):1.5f}")
print(f"________FIN SOFTMAX________" )
end = time.time()
print(f"\n\tDuración {(end - start):1.5f} s\n")
#endregion