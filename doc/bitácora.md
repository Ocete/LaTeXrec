---
title: "Generación de \\LaTeX{} a partir de imágenes de fórmulas"
subtitle: "Notas acerca del desarrollo del proyecto"
author: "José Antonio Álvarez Ocete, Daniel Pozo Escalona"

titlepage: true
reference-section-title: Referencias
graphics: true
lang: es
---

La numeración de los apartados hace referencia a la numeración de la
*propuesta.md*.

# 1. Eliminación de ambigüedades

Como ya discutimos en la propuesta, en este problema encontramos ambigüedad en
la salida de los datos. Distintas expresiones $LaTeX$ generar la misma salida o
una casi indistiguible en la imagen: por ejemplo, `\sin` y `\operatorname{sin}`.
Esto hace que el aprendizaje sea  más difícil. En esta primera sección
eliminamos ciertas ambigüedades. Para ello, utilizamos el [*tokenizer* de
TensorFlow](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer)
para generar una lista de tokens de las primeras 60.000 imagénes en la que
basarnos. Estudiando esta lista hemos decidido modificar las siguientes
ambigüedades, buscando un balance entre implementación sencilla y que merezca la
pena porque sean lo suficientemente relevantes:

- Los operadores con nombre más famosos (como seno, coseno, máximo y mínimo),
  tienen un comando particular en $LaTeX$: `\sin`, y `\max`. Utilizaremos estos
  comandos en vez de los respectivo `\operatorname { sin }` y
  `\operatorname { max }`. Aplicaremos está misma sustitución para los comandos
  con un asterisco: `\operatorname* { sin }`. La lista completa de operadores a
  la que le aplicamos essta sustitución es la siguiente: `\sin, \cos, \tan,
  \arcsin, \arccos, \arctan, \sinh, \cosh, \tanh, \max, \min, \exp, \log, \ln,
  \sup, \inf, \lim, \dim, \deg, \ker, \cot, \Pr, \lg, \arg, \det, \vol`.

- El comando `\prime` se utiliza en $LateX$ para mostar una comilla grande. En
  caso de utilizarse como exponente (`^ { \prime }`) tiene la misma
  representación gráfica que una comilla simple: `'`. Sustituiremos la expresión
  `^ { \prime }` por `'`.
- Los símbolos de llaves `{` y `}` se pueden escribir también como `\lbrace` y
  `rbrace` respectivamente. Los sustituiremos por la versión más corta que es,
  además, más general: `{` y `}` se pueden utilizar tanto en modo tecto como en
  modo matemáticas.
- El símbolo de la daga se puede escribir en $LaTex$ utilizando
  tanto `\dagger` como `\dag`. Aunque este símbolo apenas aparece en nuestras
  fórmulas, añadir esta sustitución es una línea extra que no añade complejidad
  ninguna. Utilizaremos su versión más corta.
- Finalmente, la tipografía aplicada al utilizar `\cal` y `\mathcal` es
  exactamente la misma aunque su sintaxis es distinta. `\cal` se utiliza para
  palabras o letras sueltas: `\cal A`; mientras que `\mathcal` se utiliza para
  expresiones más complejas: `\mathcal { sin ^ {2} ( x ) }`. Puesto que el uso
  de `\cal` está deprecado y `\mathcal` es más general, utilizaremos este
  último, reajustando la sintaxis conforme sea necesario.

# 2. Atención eficiente

Uno de los principales escollos que hemos encontrado ya desde el trabajo
preliminar en este proyecto es el enorme consumo de memoria del modelo
*transformer*. Esto se debe a que, en cada capa de atención, se ha de calcular
la matriz de atención, de tamaño cuadrático en la longitud de la secuencia
procesada. En el caso de las características provenientes de la imagen, esta
secuencia puede alcanzar longitud 500.

Recientemente, se han propuesto múltiples alternativas para aproximar el
mecanismo de atención de forma más eficiente (@tay2020efficient). En la tabla 1
de este artículo de revisión se encuentran listadas todas las alternativas,
junto con algunas características de las mismas.

Hemos seleccionado, de entre las que permiten emplear un decodificador, la
atención rápida mediante el mecanismo FAVOR+ (@choromanski_rethinking_2020).

# 3. Lectura eficiente

Una vez saltamos al dataset Im2latex encontramos el gran problema de que este
no cabia por completo en memoria. De hecho, no cabía ni la mitad. De cara a
implementar una solución escalable y eficiente a este problema creamos una clase
LaTeXrecDataset que hereda de Dataset, de tensorflow. Esta clase nos permite
leer las imágenes conformen nos hacen falta y, al mismo tiempo, quitarlas de
memoria cuando dejan de hacerlo. Todo esto se procesa de forma automática y
muy comodamente, además de permitirnos implementar 'prefetching'. Esto es,
lectura anticipada de los datos antes de que hagan falta para accelerar el
proceso. Todo esto está implementado en el archivo `datasets.py`.

# 4. Añadir logging

Añadimos un sencillo sistema de logging a distintos archivos. Esto hace que
el trabajo con paperspace sea mucho más rápido en los posteriores experimentos.
Dicho sistema se puede encontrar en el archivo `log.py`.

# 5. Implementación de codificación posicional 2-dimensional



# 6. Implementar métrica BLEU

Hemos añadido la [métrica BLEU](https://en.wikipedia.org/wiki/BLEU) para
evaluar de forma más intuitiva los resultados obtenidos. Esta métrica nos
da una estimación de cuanto se parecen dos cadenas de tokens.

Para ello nos hemos basado en
[tensor2tensor](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/bleu_hook.py#L132). Nuestra implementación se encuentra en `bleu.py`.