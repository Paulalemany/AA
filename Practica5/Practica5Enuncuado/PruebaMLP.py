from MLP_Complete import MLP_Complete
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from Utils import one_hot_encoding, accuracy
from imblearn.over_sampling import SMOTE
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4" # para que no me salga un warning en el knn loool

df = pd.read_csv("./PartidasGanadas.csv")
# print("VALORES: ", df["action"].value_counts())
# print("CORRELACION:", df.corr())
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

#balancearlos? lol estoy probando cosas
sm = SMOTE(random_state=9)
X_train, y_train = sm.fit_resample(X_train, y_train)

alpha = 0.001
num_ite = 500 
lambda_ = 1e-4
n_hidden_neurons = 32
#endregion

#region --- MLP SKLEARN ---
# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
mlp_skl = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation='logistic',           
    alpha=lambda_,                   
    learning_rate_init=alpha,
    max_iter=num_ite,                
    random_state=69
    )

mlp_skl.fit(X_train, y_train)
y_pred_sklearn = mlp_skl.predict(X_test)
acc_sklearn = accuracy_score(y_test, y_pred_sklearn) #precision
print(f"SKLEARN LOGISTIC MLP accuracy for lambda = {(lambda_):1.5f} : {(acc_sklearn):1.5f}")
#endregion

#region --- MLP SKLEARN CACHARREANDO ---
# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
mlp_skl = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation='relu',           
    alpha=lambda_,                   
    learning_rate_init=alpha,
    max_iter=num_ite,       
    random_state=69
    )

mlp_skl.fit(X_train, y_train)
y_pred_sklearn = mlp_skl.predict(X_test)
acc_sklearn = accuracy_score(y_test, y_pred_sklearn) #precision
print(f"SKLEARN RELUC MLP accuracy for lambda = {(lambda_):1.5f} : {(acc_sklearn):1.5f}")
#endregion


# region --- MLP NOSOTRAS :-) --- error en el compute cost ahora mismo
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
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
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


#region --- DECISION TREE ---
# https://scikit-learn.org/stable/modules/tree.html
tree = DecisionTreeClassifier(
    random_state=69
)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)
acc_tree = accuracy_score(y_test, y_pred_tree)
print(f"Decision Tree accuracy: {acc_tree:.5f}")
#endregion


#region --- RANDOM FOREST ---
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
forest = RandomForestClassifier(
    random_state=420,
    n_jobs=-1              # usar todos los cores disponibles
)
forest.fit(X_train, y_train)
y_pred_forest = forest.predict(X_test)
acc_forest = accuracy_score(y_test, y_pred_forest)
print(f"Random Forest accuracy: {acc_forest:.5f}")
#endregion


#region --- MATRICES DE CONFUSION MÉTRICAS ---
cfm_mlp_sk = confusion_matrix(y_test, y_pred_sklearn);
# cfm_mlp_complete = confusion_matrix(y_test, y_pred) # la nuestra
cfm_knn = confusion_matrix(y_test, y_pred_knn)
cfm_tree = confusion_matrix(y_test, y_pred_tree)
cfm_forest = confusion_matrix(y_test, y_pred_forest)

print("SKLEARN MLP Confusion Matrix:\n", cfm_mlp_sk)
# print("MLP COMPLETE Confusion Matrix:\n", cfm_mlp_complete)
print("KNN Confusion Matrix:\n", cfm_knn)
print("DECISION TREE Confusion Matrix:\n", cfm_tree)
print("RANDOM FOREST Confusion Matrix:\n", cfm_forest)
#endregion
