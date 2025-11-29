import pandas as pd
import glob

# todos los CSV que empiecen por "TankTraining"
file_list = sorted(glob.glob("./Partidas Ganadas/*.csv"))
df_list = []

for i, f in enumerate(file_list):
    # leer siguiente csv
    temp_df = pd.read_csv(f)

    # si no es el primero, eliminar la cabecera
    if i > 0:
        temp_df = temp_df.iloc[1:]

    # eliminar "win"
    temp_df = temp_df.iloc[:-1]

    df_list.append(temp_df)

# concatenacion
df_clean = pd.concat(df_list, ignore_index=True)

# eliminar columnas
# MÁS LIMPIEZA AQUÍ
# columns_to_drop = ["COMMAND_CENTER_X", "COMMAND_CENTER_Y"]
# df_clean = df_clean.drop(columns=columns_to_drop)

# filtrar filas con valores incorrectos?
#df_clean = df_clean[(df_clean["AGENT_1_X"] <= 50) & (df_clean["AGENT_2_X"] <= 50)]

# Guardar CSV limpio
df_clean.to_csv("./Partidas Ganadas/PartidasGanadas.csv", index=False)

print("Dataset limpiado, número de filas: ", len(df_clean))
