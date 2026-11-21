# Recorrido de la Etapa 3

Guía para entender `notebooks/ETAPA_3.ipynb` sección por sección: qué se hizo, por qué, y qué concepto hay detrás. Las cifras son las de la corrida local del 21/11 (5 semillas, media ± desvío en Dev).

---

## 0. Punto de partida

- Ganador de la Etapa 2: MLP 128-64-32 con Dropout 0,05 sobre el one-hot de los últimos $K = 5$ caracteres, **48,71 % ± 0,52 %**.
- Diagnóstico heredado: brecha Train-Dev de 20,72 pp (varianza) y ninguna regularización que pase del 48-49 %.
- La Etapa 3 cambia dos cosas, de a una:
  1. la **representación** de cada carácter (sección 2);
  2. el **tipo de modelo**, de MLP a LSTM (sección 3).
- Todo lo demás queda fijo: misma partición, misma métrica, mismo learning rate, Dropout, batch y épocas del ganador de la Etapa 2. Así, una diferencia en Dev se puede atribuir al cambio y no a otra cosa. La excepción son R3 y R4, que suman el esquema de la LSTM de la clase 8 (sección 3.2).
- La comparación **pre-entrenados vs desde cero** se hace con palabras (sección 8): GloVe y Word2Vec, los que nombra la consigna, no tienen vectores de caracteres.

**Concepto: por qué 5 semillas.** El resultado de una red depende de la inicialización y del orden de los batches. Con 5 semillas medimos ese ruido (el desvío) y solo creemos una mejora si es más grande que él.

---

## 1. Configuración y datos

- Se cargan el texto limpio y la partición por bloques de la Etapa 1 (`splits_bloques.json`). Test no se carga.
- Cada ventana pasa a índices enteros (`Xi_train`, `Xi_dev`): la capa `Embedding` recibe índices, no one-hot.
- `vis` muestra el espacio y el salto de línea como `<SP>` y `<NL>` en tablas y gráficos.
- Las referencias (baseline y ganador de la Etapa 2) se leen de los JSON de las etapas anteriores, no se escriben a mano.

---

## 2. Red densa sobre embeddings (consigna 4.3.B)

### Concepto: qué es un embedding

Una capa `Embedding` es una tabla $E$ de $|V| \times d$: a cada carácter le asigna una fila. Es exactamente lo mismo que multiplicar el one-hot por una matriz:

$$\text{one-hot}(c) \cdot E = \text{fila } c \text{ de } E$$

Por eso E0 se implementa como un `Embedding` fijo con la matriz identidad: reproduce el one-hot de la Etapa 2 sin cambiar el resto del código, y da exactamente 48,71 %.

La diferencia con el one-hot: en lugar de 64 posiciones con un solo 1, cada carácter es un vector denso de $d$ números que la red puede **aprender**. Caracteres que cumplen funciones parecidas pueden terminar cerca.

### Los tres modelos

| Modelo | Qué cambia | Dev | Brecha | Lectura |
|---|---|---|---|---|
| E0 One-hot | nada (control) | 48,71 % ± 0,52 % | 20,72 pp | reproduce la Etapa 2 |
| E1 Desde cero 16 dim. | embedding chico | 47,88 % ± 0,73 % | 10,11 pp | menos capacidad, menos sobreajuste |
| E2 Desde cero 300 dim. | embedding grande | 49,51 % ± 0,81 % | 25,62 pp | algo más de accuracy, más sobreajuste |

### Cómo leer cada resultado

- **E1 contra E2 (efecto de la dimensión).** 16 dimensiones son un "cuello de botella": la red tiene menos parámetros (23.840 contra 223.776) y sobreajusta la mitad, pero también acierta menos. Es el clásico intercambio sesgo-varianza.
- **E2 contra E0.** $+0,81$ pp, del tamaño del desvío entre semillas: pasar del one-hot a un embedding aprendido mejora poco.

### Concepto: perplejidad

$$\text{perplejidad} = e^{\text{loss}}$$

donde la loss es la entropía cruzada media en Dev. Se lee como "el modelo duda como si eligiera al azar entre $N$ caracteres". Con 64 clases, una perplejidad de 7,66 significa que, en promedio, duda entre unos 8.

**Accuracy y perplejidad pueden no coincidir.** La accuracy solo mira si el primer puesto acierta. La perplejidad castiga los errores con mucha confianza. E1 tiene peor accuracy que E2 pero mejor perplejidad (6,51 contra 8,66): se equivoca más, pero cuando se equivoca no está tan seguro. E2, que sobreajusta más, está demasiado seguro de sus errores.

### Concepto: cómo se cuentan los parámetros

E2: embedding $64 \times 300 = 19.200$, más la primera densa $(5 \cdot 300) \times 128 + 128 = 192.128$, más las capas siguientes y la salida. Total: 223.776.

---

## 3. Modelo de lenguaje simple: LSTM

### Concepto: qué hace una LSTM

La red densa recibe los 5 caracteres "pegados" en un solo vector y no sabe cuál vino primero salvo por la posición. Una LSTM los lee **de a uno, en orden**, y mantiene un **estado** que resume lo leído. En cada paso, tres compuertas (valores entre 0 y 1) deciden:

- qué olvidar del estado anterior (*forget gate*);
- qué información nueva guardar (*input gate*);
- qué parte del estado mostrar como salida (*output gate*).

Los pesos son los **mismos en cada paso**. Por eso R1 (5 caracteres) y R2 (40 caracteres) tienen exactamente los mismos parámetros: 247.104. Ver más contexto no agranda el modelo.

Conteo: $4 \cdot (128 \cdot (128 + 300) + 128) = 219.648$ en la LSTM, más 19.200 del embedding y 8.256 de la salida.

### 3.1 R1 y R2 con la configuración heredada

| Modelo | Dev | Brecha | Pico de Dev (época) | Tiempo por corrida |
|---|---|---|---|---|
| R1 LSTM, 5 caracteres | 49,69 % ± 0,12 % | 28,67 pp | 50,81 % (61) | 100 s |
| R2 LSTM, 40 caracteres | 45,06 % ± 1,70 % | 27,41 pp | 48,32 % (51) | 583 s |

- R1 es el más estable entre semillas. Pero sobreajusta: su pico está en la época 61 y entrena hasta la 100.
- R2 sobreajusta tanto que termina por debajo de R1. Con 40 pasos de entrada y 12.780 ventanas, la LSTM memoriza Train.
- La configuración (learning rate fijo, 100 épocas, sin Early Stopping) se había elegido para el MLP, no para una LSTM. Por eso no descartamos el contexto largo sin antes corregir cuándo parar.
- **Caché de R2:** R2 tarda unos 50 minutos con las 5 semillas. La primera corrida guarda sus resultados y curvas en `data/processed/cache_R2.json` y las siguientes los cargan, salvo que cambie su configuración. Con semillas fijas, los números son los mismos.

### 3.2 La LSTM de 40 con Early Stopping (R3)

**Concepto: el esquema de la LSTM de la clase 8.**

- *Adam con `clipnorm=1.0`* (*gradient clipping*): si la norma del gradiente supera 1, se recorta. Protege a la red recurrente de los gradientes explosivos.
- *Early Stopping*: corta si la pérdida en Dev no mejora durante 5 épocas y **recupera los pesos de la mejor época**.
- *Learning rate*: el elegido en la Etapa 2 (0,0005), el mismo para R3 y R4. El 0,005 de la clase 8 también se probó (está en `docs/EXPERIMENTOS_RETIRADOS.md`).

| Modelo | Dev | Brecha | Épocas | Tiempo por corrida |
|---|---|---|---|---|
| R2 | 45,06 % ± 1,70 % | 27,41 pp | 100 | 583 s |
| R3 = R2 + esquema de la clase 8 | 47,76 % ± 0,55 % | 9,20 pp | 53 | 284 s |

- Cortar a tiempo recupera parte: +2,70 pp sobre R2, con un desvío tres veces menor; la brecha baja de 27 a 9 pp y la corrida tarda la mitad.
- Pero R3 queda 1,93 pp debajo de R1. Como no se entrenaron igual, todavía no se sabe si es por el contexto o por el esquema.

### 3.3 Control: la LSTM de 5 con el mismo esquema (R4)

| Modelo | Dev | Peor semilla | Brecha | Perplejidad | Épocas | Tiempo por corrida |
|---|---|---|---|---|---|---|
| R1 | 49,69 % ± 0,12 % | 49,51 % | 28,67 pp | 7,64 | 100 | 100 s |
| R4 = R1 + esquema de la clase 8 | 50,07 % ± 0,54 % | 49,34 % | 10,43 pp | 5,91 | 47 | 45 s |
| R3 | 47,76 % ± 0,55 % | 47,22 % | 9,20 pp | 6,16 | 53 | 284 s |

- **R4 es R1 cortado a tiempo.** Mismo modelo y mismo learning rate: las curvas van casi superpuestas hasta que Early Stopping corta a R4.
- **En accuracy empatan** (+0,38 pp, menos que el desvío de R4). **En lo demás gana R4:** menos brecha, la mejor perplejidad de la etapa y la mitad del tiempo.
- **Con el mismo esquema, el contexto corto gana:** R4 supera a R3 por 2,31 pp. Con el lr de 0,005 pasa lo mismo (49,20 % contra 48,40 %).

---

## 4. Comparación con la Etapa 2

El ganador es el de mayor accuracy media en Dev entre todos los modelos de las secciones 2 y 3: **R4, con 50,07 % ± 0,54 %**. Mejora **+1,36 pp** sobre la Etapa 2 (48,71 %) y +9,24 pp sobre el baseline (40,83 %).

**Para la defensa:** la mejora sobre la Etapa 2 es apenas mayor que el desvío entre semillas. Lo que sí es sólido es lo que **no** funciona (R2 sin Early Stopping, el contexto largo con este volumen de datos) y por qué.

---

## 5. Ejemplos que siguen siendo difíciles (consigna 4.3.B)

Se compara el ganador con E0 (semilla 42) de tres formas.

### Por tipo de carácter

- El **espacio** es lo más fácil (89 % en E0 y 91 % en R4), porque las palabras tienen largos típicos.
- R4 y E0 casi no se diferencian por tipo: entre −1,67 y +1,36 pp.
- **Mayúsculas** (38 casos) y **puntuación** (60 casos) casi no se resuelven. Además son pocos casos: en puntuación, −1,67 pp es un acierto menos.

### Por posición dentro de la palabra

- **Posición 0** = predecir la primera letra de una palabra nueva, justo después de un espacio. Ahí los dos modelos aciertan menos del 20 %.
- Desde la segunda letra se pasa del 48 %. Adivinar **qué palabra** viene es mucho más difícil que **completar** una ya empezada.

### Los 10 errores concretos: tres patrones

1. **No saber si la palabra terminó.** En `against` y `see` apuesta a que la palabra sigue; en `mad` (de *made*) y `hand` (de *hands*), a que terminó.
2. **Varias continuaciones razonables.** En `signs th` le da 59 % a `e` (*the*) y 24 % a la real, `a` (*that*). Con ese contexto es error irreducible.
3. **Mayúsculas y comas.** Después de un punto sabe que viene una mayúscula (`T`, `I`, `A` en su top-5), pero no cuál. En `waiting` le da 87 % al espacio y 6 % a la coma.

Los patrones 1 y 3 necesitan más contexto; el 2, no. El ganador falla en el 50,00 % de Dev.

---

## 6. Qué aprendieron los embeddings (PCA)

**Concepto: PCA.** Proyecta los vectores de 300 dimensiones a las 2 direcciones de mayor varianza, para poder graficarlos. La "varianza explicada" dice qué parte de la información entra en el gráfico.

- **Aprendido desde cero, E2 (13,2 % de varianza explicada):** la información está repartida en muchas dimensiones, así que el gráfico muestra poco. Aun así, las letras no se superponen: las minúsculas frecuentes se agrupan en el centro y las mayúsculas y los signos quedan más dispersos.

**Limitación:** con 13,2 % de varianza explicada, la PCA es solo orientativa.

---

## 7. Conclusiones y experimentos fallidos

Están en la sección 7 del notebook y en la tabla 7.1. La consigna pide documentar lo que no funcionó: por eso E1, R2, R3 y el Early Stopping de la sección 8 están ahí con el motivo. Lo que se sacó del notebook (ReduceLROnPlateau, embeddings de caracteres pre-entrenados, variantes de R3, lr 0,005) está en `docs/EXPERIMENTOS_RETIRADOS.md`.

---

## 8. Embeddings de palabras: pre-entrenados vs desde cero (consigna 4.3.B)

**Por qué existe:** la consigna 4.3.B pide comparar embeddings **de palabras** pre-entrenados contra entrenados desde cero. La tarea principal es de caracteres y GloVe no tiene vectores de caracteres, así que la comparación se hace prediciendo la siguiente palabra. **No compite** con el resto: otra tarea, otra salida y otra cantidad de clases.

### Decisiones

- Contexto de 3 palabras → siguiente palabra.
- Misma partición por bloques: cada bloque se corta en palabras por separado, así no hay fuga.
- Vocabulario de Train + `<UNK>`. Una palabra de Dev que no está en Train **siempre cuenta como error**.
- GloVe 6B de 100 dimensiones.

### Resultados

| Modelo | Dev top-1 | Dev top-5 |
|---|---|---|
| Siempre "the" (referencia) | 7,63 % | — |
| P0 Desde cero | 8,08 % ± 1,02 % | 23,54 % |
| P1 GloVe congelado | 10,81 % ± 2,84 % | 25,49 % |
| P2 GloVe ajustable | 11,72 % ± 2,76 % | 25,58 % |

- **Techo de 74,8 %:** el 25,2 % de las palabras de Dev no aparece en Train.
- **P0 casi no aprende:** en 4 de las 5 semillas da exactamente 7,63 %, lo mismo que predecir siempre "the". Early Stopping recupera los pesos de una época en la que solo aprendió la palabra más frecuente.
- **GloVe ayuda en promedio** (+2,7 a +3,6 pp), pero con mucho desvío: en algunas semillas también queda cerca de "the". Una palabra tiene significado propio, y GloVe lo trae de 6.000 millones de tokens. Los vecinos lo muestran: *sea* → *waters, water, fishing, shore, river* en GloVe; *sea* → *taverns, to, everything* en P0.
- **Congelar o ajustar da lo mismo** (0,91 pp de diferencia, dentro del desvío).
- **Limitación documentada:** Early Stopping corta a las 11 épocas porque mira la pérdida, y la accuracy sigue subiendo después. Las cifras subestiman a los tres modelos.

---

## Preguntas probables en la defensa

1. **¿Por qué no hay embeddings de caracteres pre-entrenados?** GloVe y Word2Vec, los que nombra la consigna, son de palabras. Los probamos con caracteres desde un repositorio externo y no aportaban; está en `docs/EXPERIMENTOS_RETIRADOS.md`.
2. **¿Por qué R1 y R2 tienen los mismos parámetros?** Porque la LSTM usa los mismos pesos en cada paso de tiempo.
3. **¿Por qué la LSTM de 40 no aprovecha el contexto largo?** Sin Early Stopping sobreajusta. Con el mismo esquema que la de 5, queda 2,31 pp debajo. La hipótesis es que con unas 12.800 ventanas no alcanza para aprender dependencias largas.
4. **¿Si R4 y R1 empatan en accuracy, por qué gana R4?** Porque tiene la mayor accuracy media, que es la métrica de la consigna, y además menos brecha, mejor perplejidad y la mitad del tiempo.
5. **¿La mejora sobre la Etapa 2 es significativa?** +1,36 pp, apenas mayor que el desvío entre semillas. Por eso se reporta con media ± desvío y peor semilla.
6. **¿Por qué el lr de la Etapa 2 y no el 0,005 de la clase 8?** Con 0,005 el Early Stopping cortaba a R4 en la época 12, poco entrenado (49,20 %). R3 y R4 usan el mismo lr para que la comparación de contextos sea pareja.
7. **¿Por qué en palabras GloVe sí ayuda?** Una palabra tiene significado propio que se transfiere desde otro corpus.
