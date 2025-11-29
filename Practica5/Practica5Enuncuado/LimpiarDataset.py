import pandas as pd
import glob

# debug: calculo de lineas totales para contrastar
file_list = sorted(glob.glob("./Partidas Ganadas/*.csv"))

total_lines = 0
for f in file_list:
    with open(f, "r", encoding="utf-8") as file:
        lines = file.readlines()
    total_lines += len(lines)
print("TOTAL DE LÍNEAS: ", total_lines)
expected_rows = total_lines - len(file_list) - (len(file_list) - 1)
print("NUMERO ESPERADO DE LINEAS TRAS LIMPIEZA: ", expected_rows)


# todos los CSV que empiecen por "TankTraining"
df_list = []

for i, f in enumerate(file_list):
    # leer siguiente csv
    temp_df = pd.read_csv(f)

    # si no es el primero, eliminar la cabecera
    # if i > 0: esto no hace falta por la cara????
    #     temp_df = temp_df.iloc[1:]

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
df_clean.to_csv("./PartidasGanadas.csv", index=False)

print("DATASET LIMPIO, número de filas: ", len(df_clean))
