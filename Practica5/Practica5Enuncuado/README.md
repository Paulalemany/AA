Hola Isma :^)



**Enlace al repo:** 

https://github.com/Paulalemany/AA



Nuestra entrega consiste en:



**Archivos:**

&nbsp;	*DibujarDataset*.py -> (Ejercicio 2) Dibuja en 2D el dataset

&nbsp;	*LimpiarDataset*.py -> (Ejercicio 3) Junta todas las partidas en un solo archivo y quita columnas innecesarias del dataset

&nbsp;	*EntrenarModelos*.py -> (Ejercicio 4) Normaliza y entrena nuestro MLP, el de SKLearn, KNN, Decision Tree y Random Forest y exporta el de Sklearn.

&nbsp;	MLP\_Complete.py -> Nuestra implementación del MLP.

&nbsp;	ohe\_categories.json -> JSon con los headers de las categorías que son OHE para configuar con él el componente MLAgent.

&nbsp;	PartidasGanadasFacil.csv -> Nuestro dataset pasado por LimpiarDataset.py versión fácil

&nbsp;	PartidasGanadasNormal.csv -> Nuestro dataset pasado por LimpiarDataset.py versión normal



**Carpetas:**

&nbsp;	./*ModeloExportado* -> Al final de EntrenarDataset.py se exporta el modelo de SKlearn en todos los formatos y lo dejamos aquí, para luego llevarlo a la carpeta en el proyecto de Unity en el repositorio, en BattleCity\\Assets\\BattleCity\\source\\BC\\ML\\ModeloExportado

&nbsp;	./PartidasGanadasFacil -> todos los .csv de 200 partidas ganadas en fácil 

&nbsp;	./PartidasGanadasNormal -> todos los .csv de 150 partidas ganadas en normal (empezamos usando esto pero era muy poco preciso y nos cambiamos a fácil x\_x)



**Repositorio:**

&nbsp;	Hemos implementado los métodos necesarios en MLAgent, OHE, MLP etc y configurado el componente MLAgent en la escena para que use nuestro modelo, los índices que quitamos etc. 







En el ejercicio 4, las precisiones que nos ha dado cada modelo han sido:



SKLearn Logistic: 0.86680

SKLearn Reluc y cacharreando: 0.85946

Nuestro MLP: 0.80476

KNN: 0.85845

Decision Tree: 0.87617

Random Forest: 0.89820



El que mayor precisión nos ha dado es el Random Forest siempre a lo largo de todo el desarrollo de la práctica y tiene sentido porque nuestro volumen de datos (200 partidas) no es gran cosa y las acciones no están balanceadas, además de que es un modelo que está pensado específicamente para datos heterogéneos, así que tendría sentido probar a implementarlo :-)





Muchas gracias✨



Paula Alemany

Cynthia Tristán

