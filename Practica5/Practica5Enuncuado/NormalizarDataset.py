import pandas as pd
import glob
# https://docs.python.org/3/library/glob.html
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

# --- LECTURA --- 
# todos los CSV que empiecen por "TankTraining"
df_clean     = pd.read_csv("./PartidasGanadas.csv")


# !!! --- NORMALIZACIÓN --- !!!
# ! ONEHOT ENCODIGNG -> NEIGHBOURGHS 
# ! STANDARDSCALING AL RESTO DE ATRIBUTOS (NUMÉRICOS CONTINUOS)
# ! LABELENCODER -> ACTION ?
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html

ohe_columns = [
    "NEIGHBORHOOD_UP",
    "NEIGHBORHOOD_DOWN",
    "NEIGHBORHOOD_RIGHT",
    "NEIGHBORHOOD_LEFT"
]

stsc_columns = [
    "NEIGHBORHOOD_DIST_UP",
    "NEIGHBORHOOD_DIST_DOWN",
    "NEIGHBORHOOD_DIST_RIGHT",
    "NEIGHBORHOOD_DIST_LEFT",
    "AGENT_1_X", "AGENT_1_Y",
    "AGENT_2_X", "AGENT_2_Y",
    "EXIT_X", "EXIT_Y",
    "time"
]

label_column = "action"

#https://www.educative.io/answers/the-fit-vs-fittransform-methods-in-scikit-learn
# --- ONE-HOT ENCODING ---
encoder = OneHotEncoder(sparse_output=False)
ohe_data = encoder.fit_transform(df_clean[ohe_columns])

ohe_df = pd.DataFrame(
    ohe_data,
    columns=encoder.get_feature_names_out(ohe_columns), # esto hace una columna con cada posibilidad de cada atributo para que quepa todo al hacer OHE
    index=df_clean.index
)
# df_clean[ohe_columns] = ohe_df esto ya no se puede hacer pq ahora hay más columnas
df_sin_ohe = df_clean.drop(columns=ohe_columns); #quitamos los atributos orignales
df_clean = pd.concat([df_sin_ohe, ohe_df], axis=1) #los volvemos a concatenar al final ahora que están OHE


# --- STANDARD SCALING ---
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_clean[stsc_columns])
scaled_df = pd.DataFrame(
    scaled_data,
    columns=stsc_columns,
    index=df_clean.index
)
df_clean[stsc_columns] = scaled_df


# --- LABEL ENCODING ---
encoder = LabelEncoder()
df_clean[label_column] = encoder.fit_transform(df_clean[label_column])

# --- GUARDAR ---
df_clean.to_csv("./DatasetNormalizado.csv", index=False)

print("Normalización completada. Líneas:", len(df_clean))

