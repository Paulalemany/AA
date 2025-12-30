###### **RNA:**

*Ventajas:*

* Tolerancia a fallos (ruido de los datos de entrada)
* Modelos escalables
* Pueden resolver problemas no lineales

*Inconvenientes:*

* Coste computacional elevado
* Algoritmo de caja negra.



###### **KNN:**

*Ventajas:*

* No paramétrico (salvo que usemos distancias ponderadas). No hace suposiciones explícitas sobre la forma de los datos.
* Algoritmo simple tanto de explicar como de interpretar.
* Alta precisión (relativa). Es bastante alta, aunque no superior a otros modelos más sofisticados. Pero a pesar de su aparente simpleza, si se elige correctamente la distancia puede ofrecer resultados bastante buenos.
* El proceso de entrenamiento es inmediato.

*Inconvenientes:*

* Es muy sensible a los atributos irrelevantes. Hacer una buena selección de atributos relevantes es fundamental.
* Es sensible al ruido, ya que, si un ejemplo es un mal ejemplo de entrenamiento y es el seleccionado como el más similar, daremos una solución errónea. Esto se puede mitigar haciendo que K sea grande ya que reduce el ruido.
* La ejecución es lenta si hay muchos datos, ya que tiene que procesar todos los datos. Existen métodos para optimizarlo usando partición espacial (KD-tree) pero aún asi es costoso.
* Es caro en memoria ya que ocupa mucha memoria si hay muchos casos (Y tiene difícil solución, salvo limitar la memoria de trabajo). 



###### **CBR:**

*Ventajas:*

1. Razona a partir de casos previos reales.
2. Se pueden añadir nuevos casos sin reentrenar el sistema completo.
3. Buen rendimiento en dominios donde existen casos históricos similares.
4. Puede adaptarse fácilmente a cambios en el dominio.
5. Las soluciones pueden justificarse mostrando casos similares usados.

*Inconvenientes:*

1. Dependencia fuerte de la calidad y representatividad de la base de casos.
2. Alto coste en memoria si hay muchos casos almacenados.
3. La búsqueda de casos similares puede ser computacionalmente costosa.
4. Difícil definir una buena medida de similitud.
5. Sensible al ruido y a casos mal etiquetados.



###### **Árboles de Decision:**

*Ventajas:*

* El entrenamiento es muy rápido
* Es fácil de interpretar los resultados por un humano, es un algoritmo de caja blanca.
* Para algunos problemas consigue una buena precisión.
* Se pueden convertir fácilmente en reglas.
* No requiere una preparación de los datos demasiado exigente.
* Puede trabajar con variables cualitativas y cuantitativas.

*Inconvenientes:*

* Es muy dependiente al ruido de la entrada.
* Los árbol de decisión tienden al sobre-entrenamiento: minimizarlo fijando una profundidad (mas sesgo, menos varianza).
* No se puede garantizar que el árbol generado sea el óptimo.
* Hay conceptos que no son fácilmente aprendibles pues los árboles de decisión ya que las particiones del espacio de soluciones que puede hacer son aquellas que son representables mediante una sucesión de hiperplanos. Si no hay una aproximación lineal al problema, puede que den un modelo poco efectivo.
* Se recomienda balancear el conjunto de datos antes de entrenar. 



###### **Random Forest:**

*Ventajas:*

* Generalmente genera resultados muy buenos.
* Fácil de calcular.
* Dar estimaciones de qué variables son importantes para clasificar.

*Inconvenientes:*

* Sobreajusta si hay mucho ruido.
* Es más difícil de interpretar que el DT. 



###### **Support Vector Machine:**

*Ventajas:*

* Eficaz en espacios de grandes dimensiones.
* Todavía eficaz en casos donde el número de dimensiones es mayor que el número de muestras.
* Utiliza un subconjunto de puntos de entrenamiento en la función de decisión (llamada vectores de soporte), por lo que también es eficiente en memoria.
* Versátil: se pueden especificar diferentes funciones del núcleo para la función de decisión. Se proporcionan kernels comunes, pero también es posible especificar kernels personalizados.

*Inconvenientes:*

* Si el número de características es mucho mayor que el número de muestras evite el exceso de ajuste al elegir las funciones del Kernel y el término de regularización es crucial.
* Los SVMs no proporcionan directamente estimaciones de probabilidad, éstas se calculan utilizando una validación cruzada.



###### **Naive Bayes:**

*Ventajas:*

1. Muy rápido tanto en entrenamiento como en predicción.
2. Funciona bien con conjuntos de datos grandes.
3. Requiere pocos datos de entrenamiento.
4. Muy eficiente en memoria.
5. Funciona especialmente bien en clasificación de texto (spam, análisis de sentimiento).
6. Poco sensible a atributos irrelevantes.

*Inconvenientes:*

1. *Asume independencia entre las características, lo cual rara vez se cumple en la práctica.*
2. *Su rendimiento disminuye cuando las variables están fuertemente correlacionadas.*
3. *Modela mal relaciones complejas y no lineales.*
4. *Las probabilidades estimadas pueden no ser realistas.*
5. *Problemas con valores nunca vistos (aunque se mitiga con suavizado de Laplace).*











