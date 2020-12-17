# Modelo general

El modelo que proponemos es el siguiente:

- un codificador, formado por una red convolucional, seguido por
  un modelo transformer,

- un decodificador transformer, que emplea atención multi-cabezal
  sobre la salida del codificador y sobre una secuencia
  potencialmente incompleta, para predecir el siguiente token de
  la fórmula. Es decir, es un modelo de lenguaje condicional que
  modela `p(x_t | V, x_1, ..., x_{t-1})`, donde `V` es la salida del
  codificador.

# Bases de datos

Las bases de datos para el problema son:

- [TODO: FALTA INSERTAR LA DE 65K]
- im2latex-100k: https://zenodo.org/record/56198#.V2px0jXT6eA
- imlatex-170k: https://www.kaggle.com/rvente/im2latex170k

# Modelo inicial propuesto

El modelo inicial utiliza una CCN [TODO: COMPLETAR] para obtener
las características de las imágenes de entrada en un vector de
entrada para el modelo codificador - decodificador.

Utilizamos el [Tokenizer de TensorFlow](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer)
para obtener el alfabeto de entrada. Por lo tanto, este no es
fijo sino que depende de la entrada.

En cuanto a los datos, utilizamos 
[TODO: completar con la base de datos utilizada + una explicación
de por qué reducimos la entrada sobre las 65k si es que lo hacemos].

# Vías de trabajo

Dividiremos este trabajo en dos etapas. En la primera la base de datos 
elegida queda fijada e iteramos sobre los datos y el modelo para
mejorarlo todo lo posible. 

En la segunda, aplicamos el modelo obtenido a una nueva base de datos,
buscando ver cuánto hemos podido generalizar. Buscaremos reajustar 
el modelo para adecuarlo a este conjunto de datos mayor.

A continuación proponemos de forma ordenada las vías de trabajo que
seguiremos durante la primera etapa:

1. Eliminar ambigüedades en los datos de training: En este tipo
   de problemas encontramos ambigüedad en la salida de los datos.
   En particular, la misma expresión puede escribirse de distintas
   formar lo que hace que su predicción exacta sea mucho más
   complicada. Regularizaremos los datos de training para que la
   salida esperada sea lo más simple posible, sin cambiar el
   significado de la misma.

2. Estudiaremos al posibildiad de utilizar otras métricas distintas
   a la mera comparación elemento a elemento con el objetivo de evaluar
   la cercanía entre la predicción y la salida esperada. Por ejemplo,
   si la salida esperada es `[ sin , x ]`, las predicciones `[ sin, y ]`
   y `[a, +,  b, =, c ]` son ambas incorrectas, pero la primera es
   más cercana a ser correcta.

3. El cuello de botella del modelo actual es el consumo de memoria de 
   la atención multi-cabezal. Para paliar este hecho se estudiarán
   dos alternativas:
   
  3.1. Implementar alguna de las soluciones que se han propuesto para
       reducir dicho consumo de memoria [4]. 
  3.2. Si esto no fuera posible o viable, estudiaríamos cómo reducir
       la salida de nuestra CNN para que la entrada del codificador
       sea reducida. Al mismo tiempo estaríamos compactando la
       información extraido por la CNN, lo que puede acabar
       obteniendo peores resultados. Es por ello que evitaremos esta
       opción en la medida de lo posible.
  
4. Evaluaremos distintas arquitecturas para la red convolucional del
   codificador. Esta red se puede preentrenar de forma auto
   supervisada como se hace con un autoencoder. Del mismo modo
   buscaremos si es posible alguna red preexistente de este tipo para
   realizar fine tunning sobre la misma.

5. Ajustaremos los hiperparámetros de la arquitectura (número de
  capas, cabezales, tamaño de la representación, tasas de dropout).

6. Estudiaremos la posibilidad de añadir BN a las capas intermedias
   del transformer.

7. Exploraremos alternativas para el optimizador (estamos usando Adam
   con las recomendaciones del artículo en el que se propone la
   arquitectura transformer).

8. Estudiaremos la posibilidad de usar Beam Search a la hora de la
    evaluación del modelo, para obtener secuencias a las que el
    modelo asigne la máxima probabilidad.

# Referencias

Las referencias en las que nos inspiramos son fundamentalmente:

- Image-to-Markup Generation with Coarse-to-Fine Attention [1]:
  en este artículo se emplea un enfoque parecido al que
  planteamos, pero se usan modelos recurrentes en lugar de
  transformers.

- Transformer model for language understanding [2]: esta es una
  guía de la documentación de Tensorflow en la que se implementa
  un modelo codificador-decodificador con transformers. Es en la
  que nos hemos basado para la implementación preliminar.

- Transformers from scratch [3]: es una guía sobre la
  arquitectura transformer que proporciona un buen entendimiento
  de los mecanismos de atención.

[1] https://arxiv.org/abs/1609.04938
[2] https://www.tensorflow.org/tutorials/text/transformer
[3] http://peterbloem.nl/blog/transformers
[4] https://arxiv.org/abs/2009.06732
