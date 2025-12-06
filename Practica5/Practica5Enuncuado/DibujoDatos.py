import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib

# Cargamos el csv
df = pd.read_csv("Practica5\DatasetNormalizado.csv")    # Leemos los datos que tenemos
#df = pd.read_csv("Practica5Enuncuado/preprocessedData.csv")    # datos limpios de ines

# Dividimos las características y las etiquetas
X = df.drop(columns=["action"])   # Coge la primera fila con todos los títulos (Atributos) y le decimos cual es la etiqueta
y = df["action"] #.astype(int)      # Coge la columna de la etiqueta que se le pasa lo ponemos como int para que sirva de color

# Hacemos el PCA para poder pintar los datos
scaling = StandardScaler()        # Normalizamos los datos
X_scaled = scaling.fit_transform(X)    # Devuelve una matriz de X reducida al tamaño n_components

pca = PCA(n_components=2)     # Nos quedamos con 2 componentes principales
x_pca = pca.fit_transform(X_scaled)

# Mostrar varianza explicada por cada componente
print("Varianza explicada por cada componente:", pca.explained_variance_ratio_)
print("Varianza total explicada (PC1 + PC2):", pca.explained_variance_ratio_.sum())


#Lo convertimos a data frame
df_pca = pd.DataFrame({
    "PC1": x_pca[:, 0],
    "PC2": x_pca[:, 1],
    "action": y
})

#Colores para cada acción
acciones = sorted(df_pca["action"].unique())
colormap = matplotlib.colormaps.get_cmap("twilight")   # paleta con N colores
colors = colormap(np.linspace(0, 1, len(acciones)))

# Pintado de los datos
fig = plt.figure(figsize=(8,6))   # Los números son el tamaño de la imagen que va a generar en pulgadas

# Hacemos un scatter por accion
for idx, accion in enumerate(acciones):
    subset = df_pca[df_pca["action"] == accion]
    plt.scatter(
        subset["PC1"], 
        subset["PC2"],
        label=f"Acción {accion}",
        color = colors[idx],
        alpha = 0.7
        )

# Nombres de los ejes y del gráfico
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("PCA de trazas del juego")


plt.legend()
plt.show()


