import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Cargamos el csv
df = pd.read_csv("DatasetNormalizado.csv")    # Leemos los datos que tenemos
#df = pd.read_csv("Practica5Enuncuado/preprocessedData.csv")    # datos limpios de ines
#df = pd.read_csv("PartidasGanadas.csv")  

# Dividimos las características y las etiquetas
X = df.drop(columns=["action"])   # Coge la primera fila con todos los títulos (Atributos) y le decimos cual es la etiqueta
y = df["action"] #.astype(int)      # Coge la columna de la etiqueta que se le pasa lo ponemos como int para que sirva de color

# Hacemos el PCA para poder pintar los datos
scaling = StandardScaler()        # Normalizamos los datos
scaling.fit(X)
X_scaled = scaling.transform(X)    # Devuelve una matriz de X reducida al tamaño n_components

principal = PCA(n_components=2)     # Nos quedamos con 2 componentes principales
x = principal.fit_transform(X_scaled)

# Pintado de los datos
plt.figure(figsize=(8,6))   # Los números son el tamaño de la imagen que va a generar en pulgadas


scatter = plt.scatter(      # Esto es lo que realmente pinta
    x[:, 0],
    x[:, 1],
    c = y.astype(int),                  # Cada clase tiene un color diferente (c -> color, le asignamos la y para que sea diferente cada vez)
    #cmap = "tab10",         # Paleta de colores a la hora de pintar
    #cmap = "Purples",
    #cmap = "pink",
    cmap = "twilight",
    alpha = 0.7             # Transparencia de los puntos
)

# Nombres de los ejes y del gráfico
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("PCA de trazas del juego")


plt.colorbar(scatter, label="Clase (action)")
plt.show()


