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
  tienen un comando particular en $LaTeX$: `\sin`, `\cos`, `\max` y `\min`.
  Utilizaremos estos comandos en vez del respectivo `\operatorname { sin }`.
  Aplicaremos esta misma sustitución para otros comandos: `\tan`, `\arcsin`,
  `\arccos` y `\arctan`.
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

# 3. Atención eficiente

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
