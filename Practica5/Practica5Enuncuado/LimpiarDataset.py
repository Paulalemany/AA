import pandas as pd
import glob
# https://docs.python.org/3/library/glob.html
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

# --- LECTURA --- 
# todos los CSV que empiecen por "TankTraining"
file_list = sorted(glob.glob("./Practica5/Partidas Ganadas/TankTraining*.csv"))

#  --- CÁLCULOS PRE LIMPIEZA ---
total_lines = 0
for f in file_list:
    with open(f, "r", encoding="utf-8") as file:
        total_lines += len(file.readlines())
print("TOTAL DE LÍNEAS:\n", total_lines)
#las líneas esperadas son las totales menos:
# una línea por archivo (la de win)
# una línea por todos los archivos incluido el primero pq pandas ya la descarta al leerla (la de la cabecera)
expected_rows = total_lines - len(file_list)
expected_rows -= len(file_list) - 1
print("NUMERO ESPERADO DE LÍNEAS TRAS LIMPIEZA:\n", expected_rows)

# --- JUNTAR TODOS LOS DATOS ---
df_list = []
for i, f in enumerate(file_list):
    temp_df = pd.read_csv(f) # leer siguiente csv
    temp_df = temp_df.iloc[:-1] # eliminar "win"
    df_list.append(temp_df)
# concatenacion
df_clean = pd.concat(df_list, ignore_index=True)

# --- LIMPIEZA ---
# eliminar columnas
columns_to_drop = [
    "COMMAND_CENTER_X", 
    "COMMAND_CENTER_Y", 
    "CAN_FIRE", 
    "LIFE_X", 
    "LIFE_Y", 
    "HEALTH"
    ]
df_clean = df_clean.drop(columns=columns_to_drop)

# filtrar filas con valores incorrectos?
# df_clean = df_clean[(df_clean["AGENT_1_X"] <= 50) & (df_clean["AGENT_2_X"] <= 50)]

# --- GUARDAR RESULTADO ---
df_clean.to_csv("./PartidasGanadas.csv", index=False)
print("DATASET LIMPIO, número de líneas:\n", len(df_clean) + 1)  #esto no tiene en cuenta la de encabezado así que le sumamos una
#df_clean = pd.read_csv("Practica5Enuncuado/preprocessedData.csv")  #Prueba con los datos de inés

# !!! --- NORMALIZACIÓN --- !!!
# ! ONEHOT ENCODIGNG -> NEIGHBOURGHS 
# ! STANDARDSCALING AL RESTO DE ATRIBUTOS (NUMÉRICOS CONTINUOS)
# ! LABELENCODER -> ACTION ?
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html

# Variables categóricas
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
# df_clean[ohe_columns] = ohe_df # esto ya no se puede hacer pq ahora hay más columnas
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

# --- GUARDAR --- lo hago por ver enel datawrangler si lo he hecho bien
df_clean.to_csv("./DatasetNormalizado.csv", index=False)