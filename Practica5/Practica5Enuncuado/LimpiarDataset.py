import pandas as pd
import glob
# https://docs.python.org/3/library/glob.html
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- LECTURA --- 
# todos los CSV que empiecen por "TankTraining"
file_list = sorted(glob.glob("./Partidas Ganadas/TankTraining*.csv"))

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
expected_rows -= len(file_list) - 1;
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
#df_clean = df_clean[(df_clean["AGENT_1_X"] <= 50) & (df_clean["AGENT_2_X"] <= 50)]


# !!! --- NORMALIZACIÓN --- !!!
# ! ONEHOT ENCODIGNG -> NEIGHBOURGHS Y ACTION?
# ! STANDARDSCALING AL RESTO DE ATRIBUTOS (NUMÉRICOS)
# ! LABELENCODER -> ACTION ?

ohe_columns = []
stsc_columns = []
laben_columns = []










# --- GUARDAR RESULTADO ---
df_clean.to_csv("./PartidasGanadas.csv", index=False)
print("DATASET LIMPIO, número de líneas:\n", len(df_clean) + 1)  #esto no tiene en cuenta la de encabezado así que le sumamos una