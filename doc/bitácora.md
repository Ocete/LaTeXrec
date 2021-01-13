---
title: "Generación de \\LaTeX{} a partir de imágenes de fórmulas"
subtitle: "Notas acerca del desarrollo del proyecto"
author: "José Antonio Álvarez Ocete, Daniel Pozo Escalona"

titlepage: true
reference-section-title: Referencias
graphics: true
lang: es
---

# Atención eficiente

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
