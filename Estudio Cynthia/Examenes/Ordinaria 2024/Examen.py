from MLP_Complete import MLP_Complete
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib

#LECTURA
df = pd.read_csv("./Examenes/Ordinaria 2024/dementia_dataset.csv")                      

#ej 1: LIMPIEZA
columns_to_drop = [
    "Subject ID", 
    "MRI ID", 
    "Hand", 
    ]
df = df.drop(columns=columns_to_drop)

for col in df.columns:
    print(repr(col)) #colunnas despues de la limpieza
    
#ej 2: DIBUJADO
clase = "Group"
dfd = df.drop(columns="M/F") #para dibujarlo obviaremos el atributo categórico 

x = dfd.drop(columns=[clase]) 
y = dfd[clase] 
scaling = StandardScaler()        
x_scaled = scaling.fit_transform(x)    
pca = PCA(n_components=2) 
x_pca = pca.fit_transform(x_scaled)
df_pca = pd.DataFrame({
    "PC1": x_pca[:, 0],
    "PC2": x_pca[:, 1],
    clase: y
})

grupos = sorted(df_pca[clase].unique())
colormap = matplotlib.colormaps.get_cmap("twilight")
colors = colormap(np.linspace(0, 1, len(grupos)))

fig = plt.figure(figsize=(8,6)) 
for idx, grupo in enumerate(grupos):
    subset = df_pca[df_pca[clase] == grupo]
    plt.scatter(
        subset["PC1"], 
        subset["PC2"],
        label=f"Grupo {grupo}",
        color = colors[idx],
        alpha = 0.7
        )
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("PCA Demencia")
plt.legend()
plt.show()