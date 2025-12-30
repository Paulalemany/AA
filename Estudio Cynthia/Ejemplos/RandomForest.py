import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from Utils import ExportRandomForest

df = pd.read_csv("PartidasGanadasFacil.csv")                      # Datos nuestros
#df = pd.read_csv("Practica5/dementia_dataset.csv")
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
# sm = SMOTE(random_state=9)
# X_train, y_train = sm.fit_resample(X_train, y_train)

alpha = 0.01
num_ite = 2000 
#lambda_ = 1e-4
lambda_ = 0
n_hidden_neurons = 5
#endregion

print(f"________Entrenamos________" )

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
cfm_forest = confusion_matrix(y_test, y_pred_forest)
print("RANDOM FOREST Confusion Matrix:\n", cfm_forest)
#endregion

print(f"________Exportamos________" )

ExportRandomForest(forest, "random_forest.json")
print(">>SKLearn Random Forest EXPORTED TO UNITY <<")