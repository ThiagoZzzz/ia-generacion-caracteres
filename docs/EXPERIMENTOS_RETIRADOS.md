# Experimentos retirados del notebook principal

El notebook de entrega contiene solo lo que pide la consigna. Este documento registra lo que también probamos, qué aprendimos y por qué lo sacamos. Las cifras salen de corridas reales (5 semillas, media ± desvío en Dev) y los notebooks completos quedan guardados como reserva.

| Reserva | Contenido |
|---|---|
| `notebooks/reserva_plateau/ETAPA_1_Y_2_CON_PLATEAU.ipynb` | Etapa 2 con la sección 7.3 (ReduceLROnPlateau) |
| `notebooks/reserva_plateau/ETAPA_3_CON_PLATEAU.ipynb` | Etapa 3 con Plateau, embeddings de caracteres pre-entrenados y variantes de R3, con salidas |
| `notebooks/reserva/ETAPA_3_CON_EMB_CARACTERES.ipynb` | Etapa 3 con la sección 1, ya sin Plateau (sin salidas) |
| `notebooks/reserva_plateau/*.json` | resultados de las corridas con Plateau |

---

## 1. ReduceLROnPlateau (Etapas 2 y 3)

**Qué probamos.** Bajar el learning rate cada vez que la pérdida en Dev se estanca. En la Etapa 2 (sección 7.3), con dos variantes: *suave* (×0,5 cada 3 épocas sin mejora) y *agresivo* (×0,2 cada 2 épocas). En la Etapa 3, el agresivo junto con Early Stopping para R3 y R4.

**Qué encontramos (Etapa 2, MLP).**

| Esquema | Accuracy en el mínimo de `val_loss` | Pico de accuracy | Distancia entre mínimo y pico |
|---|---|---|---|
| lr fijo 0,0005 | 45,75 % ± 0,50 % | 47,56 % | 37,4 épocas |
| Plateau suave | 46,23 % ± 0,43 % | 47,38 % | 25,0 épocas |
| Plateau agresivo | 46,76 % ± 0,66 % | 47,12 % | 1,8 épocas |

- Plateau **alinea** el mínimo de la pérdida con el pico de accuracy, que es lo que necesita Early Stopping para cortar bien.
- Pero **no sube el techo**: los picos quedan dentro del desvío entre semillas.
- Con Dropout rinde **menos** que el lr fijo: 47,80 % y 47,93 % con Dropout 0,05 y 0,1, contra 48,71 % y 48,61 %.

**En la Etapa 3,** R4 (LSTM de 5 caracteres con Plateau y ES) llegó a 50,95 % ± 0,19 %.

**Por qué lo sacamos.** No aparece en ninguna clase ni en la consigna, así que suma algo más que justificar en la defensa. En la Etapa 2 no lo usaba el ganador, y en la Etapa 3 lo reemplazamos por el esquema de la clase 8: Adam con `clipnorm=1.0` y Early Stopping con paciencia 5, con el lr fijo elegido en la Etapa 2 (ver sección 5).

---

## 2. Learning rate de 0,0001 con 300 épocas (Etapa 2)

**Qué probamos.** En la sección 7.1, el lr de 0,0001 no llegaba a converger en 100 épocas, así que le dimos 300.

**Qué encontramos.** Llega al mismo pico que 0,0005 (47,61 %), pero recién en la época 265: el mismo resultado con tres veces más entrenamiento.

**Por qué lo sacamos.** Estaba dentro de la sección 7.3 y no cambia ninguna decisión.

---

## 3. Embeddings de caracteres pre-entrenados (Etapa 3, sección 1: E3 y E4)

**Qué probamos.** Vectores de 300 dimensiones del repositorio `minimaxir/char-embeddings`, donde el vector de cada carácter es el promedio de los vectores GloVe 840B de las palabras que lo contienen. Cubrían 62 de nuestros 64 caracteres; faltaban el espacio y la raya.

**Qué encontramos.**

| Modelo | Dev | Train | Lectura |
|---|---|---|---|
| E2: desde cero, 300 dim. | 49,51 % ± 0,81 % | 75,13 % | referencia |
| E3: pre-entrenado congelado | 38,29 % ± 0,80 % | 38,64 % | **sesgo alto**, por debajo del baseline |
| E4: pre-entrenado ajustable | 49,06 % ± 0,75 % | 69,88 % | empata con E2 |

- En la PCA, todas las letras de los vectores pre-entrenados formaban un único grupo compacto: al promediar miles de palabras, casi todas las letras quedan parecidas.
- Congelados, la red no podía distinguirlas. Ajustables, la red los reescribía y terminaba igual que desde cero.

**Por qué lo sacamos.** La consigna habla de embeddings **de palabras** (GloVe, Word2Vec), y ninguno trae vectores de caracteres. El archivo viene de un repositorio externo que no es material de la materia. La comparación entre pre-entrenados y desde cero queda cubierta por la sección 8, que usa GloVe con palabras, tal como pide la consigna.

---

## 4. Variantes regularizadas de la LSTM de 40 caracteres (Etapa 3, R3)

**Qué probamos.** Sobre R3 (LSTM de 40 caracteres con Plateau y ES, 48,81 % ± 0,63 %), una combinación incremental con dos candidatos. Cada uno se aceptaba solo si mejoraba más que el desvío de R3 (0,63 pp).

| Variante | Dev | Cambio | ¿Se aceptó? |
|---|---|---|---|
| R3 + Dropout 0,2 | 49,06 % ± 0,48 % | +0,25 pp | No: dentro del desvío |
| R3 + embedding de 16 dim. | 45,94 % ± 0,59 % | −2,87 pp | No: pierde accuracy también en Train (52,97 %), pasa a tener sesgo |

Estas cifras son de la corrida con Plateau; con el esquema de la clase 8 no se volvieron a correr.

**Qué aprendimos.** Regularizar más la LSTM de 40 no la acerca a la de 5 caracteres: una variante no mueve la accuracy y la otra la empeora.

**Por qué lo sacamos.** No cambian ninguna conclusión, la consigna no pide regularizar en la Etapa 3 (eso es de la Etapa 2) y costaban unos 7,5 minutos de entrenamiento cada una (5 semillas de 221 a 225 s).

---

## 5. Learning rate de 0,005 y paciencia del Early Stopping (Etapa 3, R3 y R4)

**Qué probamos.** El esquema de la clase 8 tal cual: Adam con lr fijo de 0,005, `clipnorm=1.0` y Early Stopping con paciencia 5. Después, sobre R4 (LSTM de 5 caracteres), separamos el efecto del lr y el de la paciencia.

**Qué encontramos.**

| Configuración | Dev | Peor semilla | Train | Épocas | s/corrida |
|---|---|---|---|---|---|
| R3, lr 0,005, paciencia 5 | 48,40 % ± 1,11 % | 46,91 % | 55,48 % | 18 | 105 |
| R4, lr 0,005, paciencia 5 | 49,20 % ± 0,96 % | 47,81 % | 57,81 % | 12 | 15 |
| R4, lr 0,005, paciencia 10 | 49,20 % ± 0,96 % | 47,81 % | 57,81 % | 17 | 18 |
| R4, lr 0,0005, paciencia 5 | 50,07 % ± 0,54 % | 49,34 % | 60,50 % | 47 | 52 |
| R4, lr 0,0005, paciencia 10 | 50,07 % ± 0,54 % | 49,34 % | 60,50 % | 52 | 65 |
| R3, lr 0,0005, paciencia 5 (la del notebook) | 47,76 % ± 0,55 % | 47,22 % | 56,96 % | 53 | 284 |

- **La paciencia no cambia nada:** con 5 o 10 épocas se recupera el mismo mínimo de `val_loss`, así que queda el mismo modelo. La de 10 solo tarda más.
- **El lr sí:** con 0,005 la pérdida en Dev llega a su mínimo en la época 7 u 8 y el Early Stopping corta con el modelo poco entrenado (Train 57,81 %). Con 0,0005 el mínimo llega entre las épocas 38 y 45 y la accuracy sube 0,87 pp, con la mitad de desvío entre semillas.

- **A R3 el lr nuevo no le sirve:** con 0,0005 baja de 48,40 % a 47,76 %. Con los dos lr, la LSTM de 5 caracteres supera a la de 40 (por 0,80 y 2,31 pp), así que la conclusión sobre el contexto no depende de esta elección.

**Por qué lo sacamos.** R3 y R4 quedan con el esquema de la clase 8 y el lr elegido en la Etapa 2 (0,0005): los dos tienen que entrenarse igual para comparar el contexto, y con ese lr sale el mejor modelo de la etapa. Las variantes de paciencia se corrieron fuera del notebook (copia de prueba con las 5 semillas).

---

## 6. Cambios de código para que se pueda explicar

No son experimentos; dan el mismo resultado con código más simple.

| Dónde | Antes | Ahora | Efecto en las cifras |
|---|---|---|---|
| Etapa 2, one-hot del MLP | matriz de ceros con dos `for` | `keras.utils.to_categorical` (clase 6) + `reshape` | ninguno: la matriz es idéntica (verificado) |
| Etapa 3, sección 8 | tokenizar todo el cuento y ubicar cada palabra en su bloque con `np.searchsorted` | tokenizar cada bloque por separado | mínimo: de 4.057 a 4.087 tokens (30 palabras quedan partidas en un borde) |
| Etapa 3, sección 8 | top-k en una línea | la misma cuenta en tres pasos comentados | ninguno |
