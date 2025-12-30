from MLP_Complete import MLP_Complete
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import time
from sklearn.tree import DecisionTreeClassifier
from Utils import one_hot_encoding

#LECTURA
df = pd.read_csv("./Examenes/Ordinaria 2024/dementia_dataset.csv")                      

#ej 1: LIMPIEZA
columns_to_drop = [
    "Subject ID", 
    "MRI ID", 
    "Hand", 
    ]
df = df.drop(columns=columns_to_drop)

# for col in df.columns:
#     print(repr(col)) #colunnas despues de la limpieza
    
#ej 2: DIBUJADO
clase = "Group"
dfd = df.drop(columns="M/F") #para dibujarlo obviaremos el atributo categórico 

x = dfd.drop(columns=[clase]) 
y = dfd[clase] 
scaling = StandardScaler()        
x_scaled = scaling.fit_transform(x)    
pca = PCA(n_components=2) 
x_pca = pca.fit_transform(x_scaled)
df_pca = pd.DataFrame({
    "PC1": x_pca[:, 0],
    "PC2": x_pca[:, 1],
    clase: y
})

grupos = sorted(df_pca[clase].unique())
colormap = matplotlib.colormaps.get_cmap("prism")
colors = colormap(np.linspace(0, 1, len(grupos)))

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
plt.title("PCA Demencia")
plt.legend()
# plt.show()


lda = LinearDiscriminantAnalysis(n_components=2)
x_lda = lda.fit_transform(x_scaled, y)

df_lda = pd.DataFrame({
    "LD1": x_lda[:, 0],
    "LD2": x_lda[:, 1],
    "Group": y
})


#region PREPROCESADO:
ohe_columns = ["M/F"]
sc_columns = df.drop(columns=ohe_columns + [clase]).columns
# print("OHE:", ohe_columns)
# print("Scaled:", sc_columns.tolist())

# OHE:
ohe = OneHotEncoder(sparse_output=False)
ohe_data = ohe.fit_transform(df[ohe_columns])
ohe_feature_names = ohe.get_feature_names_out(ohe_columns)
df_ohe = pd.DataFrame(ohe_data, columns=ohe_feature_names, index=df.index)

# STANDARD SCALER
scaler = StandardScaler()
sc_data = scaler.fit_transform(df[sc_columns])
df_sc = pd.DataFrame(sc_data, columns=sc_columns, index=df.index)

x = pd.concat([df_ohe, df_sc], axis=1) 

# LABEL ENCODER
le = LabelEncoder() 
y = le.fit_transform(df[clase])

# DATOS!!!
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=69)
                                                    #, stratify=y)

#endregion



#region ej 3: MLP
print(f"________MLP 3 CAPAS________" )
start = time.time()
y_train_encoded = one_hot_encoding(y_train)

x_train_np = x_train.to_numpy()
x_test_np = x_test.to_numpy()

alpha = 0.5
num_ite = 2000 
lambda_ = 0.5
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
print("MLP Confusion Matrix:\n", cfm_mlp_complete)
print(f"________FIN MLP 3 CAPAS________" )
end = time.time()
print(f"\n\tDuración {(end - start):1.5f} s\n")



print(f"________MLP 1 CAPA________" )
start = time.time()
y_train_encoded = one_hot_encoding(y_train)

x_train_np = x_train.to_numpy()
x_test_np = x_test.to_numpy()

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
print("MLP Confusion Matrix:\n", cfm_mlp_complete)
print(f"________FIN MLP 3 CAPAS________" )
end = time.time()
print(f"\n\tDuración {(end - start):1.5f} s\n")
#endregion



#region ej 4: Decision Tree
print(f"________DECISION TREE________" )
start = time.time()
tree = DecisionTreeClassifier(
    random_state=42,
    max_depth=8,
    min_samples_leaf=16
)
tree.fit(x_train, y_train)
y_pred_tree = tree.predict(x_test)
acc_tree = accuracy_score(y_test, y_pred_tree)
print(f"Decision Tree accuracy: {acc_tree:.5f}")
cfm_tree = confusion_matrix(y_test, y_pred_tree)
print("DECISION TREE Confusion Matrix:\n", cfm_tree)
print(f"________FIN DECISON TREE________" )
end = time.time()
print(f"\n\tDuración {(end - start):1.5f} s\n")
#endregion




#region ej 6:
print(f"________MLP BINARIO________" )
df[clase] = df[clase].replace("Converted", "Demented")

# ohe_columns = ["M/F"]
# sc_columns = df.drop(columns=ohe_columns + [clase]).columns

# ohe = OneHotEncoder(sparse_output=False)
# ohe_data = ohe.fit_transform(df[ohe_columns])
# ohe_feature_names = ohe.get_feature_names_out(ohe_columns)
# df_ohe = pd.DataFrame(ohe_data, columns=ohe_feature_names, index=df.index)

# scaler = StandardScaler()
# sc_data = scaler.fit_transform(df[sc_columns])
# df_sc = pd.DataFrame(sc_data, columns=sc_columns, index=df.index)

# x = pd.concat([df_ohe, df_sc], axis=1) 
# le = LabelEncoder() 
y = le.fit_transform(df[clase])
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=69)

start = time.time()
# y_train_encoded = one_hot_encoding(y_train)
y_train_bin = y_train.reshape(-1, 1)
y_test_bin = y_test.reshape(-1, 1)

x_train_np = x_train.to_numpy()
x_test_np = x_test.to_numpy()

mlp_complete = MLP_Complete(
    inputLayer=x_train_np.shape[1], 
    hiddenLayers=[128, 64], 
    outputLayer=1
    )
Jhistory = mlp_complete.backpropagation(x_train_np,y_train_bin,alpha,lambda_,num_ite, verbose=100)
a_list, z_list = mlp_complete.feedforward(x_test_np)
a3 = a_list[-1]   # activación de la última capa
y_pred = mlp_complete.predict_binary(a3)

acc_complete = accuracy_score(y_test_bin, y_pred) #precision¡
print(f"MLP Accuracy for Lambda = {(lambda_):1.5f} : {(acc_complete):1.5f}")
cfm_mlp_complete = confusion_matrix(y_test, y_pred) # la nuestra
print("MLP Confusion Matrix:\n", cfm_mlp_complete)
print(f"________FIN MLP BINARIO________" )
end = time.time()
print(f"\n\tDuración {(end - start):1.5f} s\n")
#endregion