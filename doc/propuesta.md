---
title: "Generación de \\LaTeX{} a partir de imágenes de fórmulas"
subtitle: "Propuesta de proyecto final"
author: "José Antonio Álvarez Ocete, Daniel Pozo Escalona"

titlepage: true
reference-section-title: Referencias
graphics: true
lang: es
---

# Modelo general

El modelo que proponemos es el siguiente:

- un codificador, formado por una red convolucional, seguido por un modelo
  *transformer*,

- un decodificador *transformer*, que emplea atención multi-cabezal sobre la
  salida del codificador y sobre una secuencia potencialmente incompleta, para
  predecir el siguiente token de la fórmula. Es decir, es un modelo de lenguaje
  condicional que modela $p(x_t | V, x_1, ..., x_{t-1})$, donde $V$ es la salida
  del codificador.
  
Utiliza una red convolucional para obtener las características de las imágenes
de entrada en un vector de entrada para el modelo codificador-decodificador.

# Bases de datos

Las bases de datos para el problema son:

- im2latex-100k (@kanervisto_anssi_2016_56198), y
- imlatex-170k (@im2latex_170k): contiene 65000 ejemplos, que se añaden a los
  100000 de im2latex-100k.

Ambas bases de datos contienen ambigüedad en los ejemplos, en forma de fórmulas
o segmentos de fórmulas que, escritos de distinta forma, producen la misma
imagen. Dedicaremos parte del tiempo del proyecto a diseñar formas de
normalizar estos datos, para mejorar los resultados.

Por ello, las evaluaciones iniciales con modelos de referencia y las finales
pueden no ser directamente comparables. Esto lo paliaremos de dos formas: con la
base de datos sintética que introducimos en la sección sobre el trabajo
preliminar, y evaluando los modelos de referencia en la base de datos
normalizada. Además, hay que tener en cuenta que lo relevante en este problema
es dar expresiones \LaTeX{} que produzcan las imágenes que se proporcionen al
modelo, más que estas coincidan con unas prefijadas.

\begin{figure}[h]
  \centering
  \includegraphics{fig/ejemplo-im2latex.pdf}
  \caption{Tres muestras de im2latex-170k.}
  \label{fig:muestras-1}
\end{figure}


# Trabajo preliminar

Hemos realizado algún trabajo preliminar para motivar propuesta y comprobar que:

- podemos implementar y ejecutar los modelos, y

- estos pueden obtener resultados no despreciables en el problema.

## Base de datos sintética

Lo primero que hemos hecho ha sido crear una base de datos similar a las que
pretendemos tratar, en la que las fórmulas han sido generadas a partir de una
gramática relativamente sencilla, que contiene una cantidad pequeña de símbolos.

Hemos generado 50.000 ejemplos distintos. Los guiones Python que hemos escrito a
tal efecto permiten cambiar la gramática y generar conjuntos de datos con
cantidades arbitrarias de ejemplos únicos.

Además, esta base de datos puede ser útil para evaluar cambios a la arquitectura
o hiperparámetros de forma menos costosa.

\begin{figure}[h]
  \centering
  \includegraphics{fig/ejemplo-sintetica.pdf}
  \caption{Muestras de la base de datos sintética.}
  \label{fig:muestras-sintetica}
\end{figure}

## Procesamiento de los datos

Para las imágenes, el procesamiento que realizamos es redimensionarlas a una
altura común, manteniendo la relación de aspecto.

Para las fórmulas, utilizamos el [*tokenizer* de
TensorFlow](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer)
para obtener el alfabeto de entrada. Por lo tanto, este no es fijo sino que
depende del conjunto de datos. Esto nos permite codificar las fórmulas
numéricamente.

Tras esto, eliminamos los ejemplos con imágenes demasiado grandes para ser
procesadas por el modelo, debido al coste cuadrático del mecanismo de atención.

## Evaluación de un modelo de referencia

Hemos definido y evaluado un modelo de referencia, tanto en la base de datos
sintética como en im2latex-170k. Dicho modelo posee un codificador convolucional
con tres capas Conv-BN-ELU-MaxPooling y una sola capa de atención en el
codificador y en el decodificador.

En la base de datos sintética, alcanza el 60 % de precisión en un conjunto de
validación, lo que sugiere que la arquitectura puede ser adecuada para el
problema.

En 30.000 ejemplos de im2latex-170k, alcanza un 15 % de precisión en un conjunto
de validación.

\begin{figure}[h]
  \centering
  
  \subfloat{
  \begin{modelblock}{Codificador convolucional}
  Conv2D-BN-ELU(64 filtros) \\
  MaxPooling2D(2$\times$2) \\
  Conv2D-BN-ELU(64 filtros) \\
  MaxPooling2D(2$\times$2) \\
  Conv2D-BN-ELU(64 filtros) \\
  MaxPooling2D(2$\times$2)
  \end{modelblock}
  }
  \subfloat{
  \begin{modelblock}{Codificador \textit{transformer}}
  Codif. conv. \\
  Codificación de posición \\
  Dropout \\
  Capa de codificador
  \end{modelblock}
  }
  \subfloat{
  \begin{modelblock}{Decodificador \textit{transformer}}
  Embedding \\
  Codificación de posición \\
  Dropout \\
  Capa de decodificador
  \end{modelblock}
  }
  
  \caption{Bloques del modelo de referencia.}
  \label{fig:bloques-referencia}
\end{figure}

# Plan de trabajo

Dividiremos este trabajo en dos etapas. En la primera la base de datos elegida
queda fijada e iteramos sobre los datos y el modelo para mejorarlo todo lo
posible.

En la segunda, aplicamos el modelo obtenido a una nueva base de datos, buscando
ver cuánto hemos podido generalizar. Buscaremos reajustar el modelo para
adecuarlo a este conjunto de datos mayor.

A continuación proponemos de forma ordenada las vías de trabajo que
seguiremos durante la primera etapa:

1. Eliminar ambigüedades en los datos de entrenamiento: en este problema
   encontramos ambigüedad en la salida de los datos. Distintas expresiones LaTeX
   pueden generar la misma salida o una casi indistiguible en la imagen: por
   ejemplo, `\sin` y `\operatorname{sin}`. Esto hace que el aprendizaje sea
   mucho más difícil. Normalizaremos los datos de entrenamiento para que la
   salida esperada sea lo más simple posible, sin cambiar el significado de la
   misma.

2. Estudiaremos la posibilidad de utilizar otras métricas distintas a la mera
   comparación elemento a elemento (por ejemplo,
   [BLEU](https://en.wikipedia.org/wiki/BLEU)) con el objetivo de evaluar la
   cercanía entre la predicción y la salida esperada. Por ejemplo, si la salida
   esperada es `[\sin , x]`, las predicciones `[y, \sin]` y `[a, +, b, =, c]`
   son ambas incorrectas, pero la primera es más cercana a ser correcta.

3. El cuello de botella del modelo actual es el consumo de memoria de 
   la atención multi-cabezal. Para paliar este hecho se estudiarán
   dos alternativas:
   
   - Implementar alguna de las soluciones (@tay2020efficient) que se han
       propuesto para reducir dicho consumo de memoria.
	   
   - Si esto no fuera posible o viable, estudiaríamos cómo reducir la salida de
       nuestra CNN para que la entrada del codificador sea reducida. Al mismo
       tiempo estaríamos compactando la información extraido por la CNN, lo que
       puede acabar obteniendo peores resultados. Por ello, puede ser preferible
       la primera opción.
  
4. Evaluaremos distintas arquitecturas para la red convolucional del
   codificador. Esta red se puede preentrenar de forma auto supervisada como se
   hace con un autoencoder. Del mismo modo buscaremos si es posible alguna red
   preexistente de este tipo para realizar un ajuste fino sobre la misma.

5. Ajustaremos los hiperparámetros de la arquitectura (número de capas,
  cabezales, tamaño de la representación, tasas de *dropout*).

6. Estudiaremos la posibilidad de añadir BN a las capas intermedias
   del transformer.

7. Exploraremos alternativas para el optimizador (estamos usando Adam
   con las recomendaciones del artículo en el que se propone la
   arquitectura transformer).

8. Estudiaremos la posibilidad de implementar codificación posicional
   dos dimensional en vez de la unidimensional actual con el objetivo
   de almacenar información no sólo sobre el orden sino también sobre
   la posición relativa de las características originales antes de aplanarlas.

9. Estudiaremos la posibilidad de usar Beam Search a la hora de la
    evaluación del modelo, para obtener secuencias a las que el
    modelo asigne la máxima probabilidad.

# Bibliografía

Las referencias en las que nos inspiramos son fundamentalmente:

- @deng2017imagetomarkup: en este artículo se emplea un enfoque parecido al que
  planteamos, pero se usan modelos recurrentes en lugar de transformers.

- Transformer model for language understanding (@tensorflow_transformer): esta
  es una guía de la documentación de Tensorflow en la que se implementa un
  modelo codificador-decodificador con transformers. Es en la que nos hemos
  basado para la implementación preliminar.

- Transformers from scratch (@bloem_transformers): es una guía sobre la
  arquitectura transformer que proporciona un buen entendimiento de los
  mecanismos de atención.
