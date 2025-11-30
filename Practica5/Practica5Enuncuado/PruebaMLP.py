from MLP_Complete import MLP_Complete
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from Utils import one_hot_encoding, accuracy



df = pd.read_csv("./PartidasGanadas.csv")

#region --- NORMALIZAR ---
ohe_columns = [
    "NEIGHBORHOOD_UP",
    "NEIGHBORHOOD_DOWN",
    "NEIGHBORHOOD_RIGHT",
    "NEIGHBORHOOD_LEFT"
    ]
sc_columns = [
    "NEIGHBORHOOD_DIST_UP",
    "NEIGHBORHOOD_DIST_DOWN",
    "NEIGHBORHOOD_DIST_RIGHT",
    "NEIGHBORHOOD_DIST_LEFT",
    "AGENT_1_X",
    "AGENT_1_Y",
    "AGENT_2_X",
    "AGENT_2_Y",
    "EXIT_X",
    "EXIT_Y",
    "time"
    ]


# ! X !
# OHE
ohe = OneHotEncoder(sparse_output=False)
ohe_data = ohe.fit_transform(df[ohe_columns])
ohe_feature_names = ohe.get_feature_names_out(ohe_columns)
df_ohe = pd.DataFrame(ohe_data, columns=ohe_feature_names, index=df.index)

# STANDARD SCALER
scaler = StandardScaler()
sc_data = scaler.fit_transform(df[sc_columns])
df_sc = pd.DataFrame(sc_data, columns=sc_columns, index=df.index)

X = pd.concat([df_ohe, df_sc], axis=1) # TODOS LOS DATOS YA NORMALIZADOS

# ! Y !
# LABEL ENCODER
le = LabelEncoder() 
y = le.fit_transform(df["action"])
#endregion 



#region --- DATOS!!! ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69, stratify=y)

alpha = 1.0
num_ite = 2000 
lambda_ = 0.0
n_hidden_neurons = 25
#endregion

#region --- MLP SKLEARN ---
mlp_skl = MLPClassifier(
    hidden_layer_sizes=(n_hidden_neurons,), 
    activation='logistic',
    alpha=lambda_, 
    learning_rate_init=alpha,
    max_iter=num_ite, 
    random_state=420
    )

mlp_skl.fit(X_train, y_train)
Y_pred_sklearn = mlp_skl.predict(X_test)
acc_sklearn = accuracy_score(y_test, Y_pred_sklearn) #precision
print(f"SKLEARN: Calculated accuracy for lambda = {(lambda_):1.5f} : {(acc_sklearn):1.5f}")
#endregion



# region --- MLP NOSOTRAS :-) ---
# y_train_encoded = one_hot_encoding(y_train)
# mlp_complete = MLP_Complete(
#     inputLayer=X_train.shape[1], 
#     hiddenLayers=[32, 16, 8], 
#     outputLayer=y_train_encoded.shape[1]
#     )
# Jhistory = mlp_complete.backpropagation(X_train,y_train_encoded,alpha,lambda_,num_ite)
# a_list, z_list = mlp_complete.feedforward(X_test)
# a3 = a_list[-1]   # activación de la última capa
# y_pred = mlp_complete.predict(a3)
# acc_complete = accuracy_score(y_test, y_pred) #precision¡
# print(f"OURS: Calculated accuracy for lambda = {(lambda_):1.5f} : {(acc_complete):1.5f}")
#endregion


#region --- KNN ---
#endregion


#region --- DECISION TREE ---
#endregion


#region --- RANDOM FOREST ---
#endregion


#region --- MATRICES DE CONFUSION / ACCURACY / MÉTRICAS ---
#endregion
