import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


#LECTURA
df = pd.read_csv("./Examenes/Ordinaria 2024/dementia_dataset.csv")                      

columns_to_drop = [
    "Subject ID", 
    "MRI ID", 
    "Hand",
    "M/F" 
    ]
df = df.drop(columns=columns_to_drop)
    
#ej 2: DIBUJADO
clase = "Group"
x = df.drop(columns=[clase]) 
y = df[clase] 
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
colormap = matplotlib.colormaps.get_cmap("prism")
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


lda = LinearDiscriminantAnalysis(n_components=2)
x_lda = lda.fit_transform(x_scaled, y)

df_lda = pd.DataFrame({
    "LD1": x_lda[:, 0],
    "LD2": x_lda[:, 1],
    "Group": y
})

for idx, grupo in enumerate(grupos):
    subset = df_lda[df_lda[clase] == grupo]
    plt.scatter(
        subset["LD1"], 
        subset["LD2"],
        label=f"Grupo {grupo}",
        color = colors[idx],
        alpha = 0.7
        )
plt.xlabel("LCA Component 1")
plt.ylabel("LCA Component 2")
plt.title("LCA Demencia")
plt.legend()
plt.show()