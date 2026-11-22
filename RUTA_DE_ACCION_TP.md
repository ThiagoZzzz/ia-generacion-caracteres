# Ruta de acción — TP Final IA 2026 (UNAB)

> Dataset: *The Festival* (H. P. Lovecraft) · Tarea: predecir el siguiente carácter dado un contexto de 40 caracteres
> Última actualización: 21/11/2026 (cifras de las corridas locales del 21/11, Python 3.12, CPU) · Entrega final según la consigna: **21/11/2026** (verificá en el Campus la fecha límite de cada etapa)

**Cómo usarlo:** tildá `[x]` cuando algo esté hecho *y* verificado en el notebook. Los ítems marcados con ❗ son riesgos o decisiones que conviene resolver antes de seguir.

---

## Estado general

| Etapa | Contenido | Obligatoria | Estado |
|---|---|---|---|
| 1 | EDA, partición, baseline | Sí | Hecha |
| 2 | Análisis de errores, MLP, regularización | Sí | Hecha (falta la revisión manual de los 50 errores) |
| 3 | Arquitectura del dominio (texto) | Sí | Hecha (incluye la comparación con GloVe de palabras) |
| 4 | VAE / GAN | **No** | No se hace (trabajo futuro) |
| 5 | Comparación, Test, ética, conclusiones | Sí | Hecha (`notebooks/ETAPA_5.ipynb`) |
| — | Formato de entrega | Sí | Pendiente |

---

## 0. Decisiones transversales (se toman una vez)

- [x] **Framework único** para todo el trabajo: TensorFlow 2.21 / Keras 3.15. Nota: TensorFlow nativo en Windows corre solo en CPU; aun así, todo corrió en local (≈ 47 min las Etapas 1–2, ≈ 95 min la 3 con R2 o ≈ 45 min con su caché, ≈ 4 min la 5).
- [x] **Semillas fijas** (`random`, `numpy`, framework) en una celda de configuración común.
- [x] **Métrica única del TP: accuracy en Dev.** Secundaria: perplejidad (= exp de la pérdida en Dev). El F1-macro no se calculó.
- [x] **Cada modelo se entrena con 5 semillas** (42, 123, 456, 789, 2026) y se reporta media ± desvío. Con 2.880 muestras en Dev, el margen de error es de ≈ ±1,8 pp: una diferencia menor no se puede defender como mejora.
- [x] **Medir parámetros entrenables y tiempo de entrenamiento en el momento** de entrenar cada modelo (en `resultados_modelos.json` y `resultados_etapa3.json`).
- [x] **Test congelado:** ninguna celda lo usa para entrenar, elegir hiperparámetros ni evaluar hasta la Etapa 5.
- [x] Los hiperparámetros se eligen **con Dev**, nunca con Test.
- [x] **Registrar los experimentos que fallan** (la cátedra los valora igual que los exitosos): secciones 8.7 (Etapa 2) y 7.1 (Etapa 3) de los notebooks y tabla del final de este archivo.
- [x] **El notebook final solo lleva lo que pide la consigna y técnicas vistas en clase.** Lo demás se guarda en `notebooks/reserva*/` y se registra en `docs/EXPERIMENTOS_RETIRADOS.md` (ReduceLROnPlateau, embeddings de caracteres pre-entrenados, variantes de R3, lr 0,005 de la clase 8).
- [ ] Commit (GitHub o Drive) al terminar cada etapa.

---

## Etapa 1 — Exploración y baseline

**4.1.1 EDA**
- [x] Carga del archivo original y limpieza reproducible
- [x] Descripción de variables (tipos, rangos, nulos)
- [x] ≥ 3 gráficos relevantes
- [x] Problemas identificados: desbalance, outliers, data mismatch (con evidencia numérica)

**4.1.2 Partición**
- [x] Train / Dev / Test por bloques contiguos (35 / 8 / 7 bloques; 12.780 / 2.880 / 2.520 muestras)
- [x] Estrategia justificada y ausencia de fuga verificada
- [x] Proporciones de clase comparadas entre conjuntos

**4.1.3 Baseline**
- [x] Preprocesamiento mínimo y codificación única carácter ↔ índice
- [x] Baseline: regresión logística sobre one-hot de los últimos 5 caracteres
- [x] Métrica única justificada y reportada en Dev: **40,83 %** (referencia trivial: 17,85 %)

**Pendientes de la Etapa 1 (necesarios para etapas posteriores)**
- [x] Guardar en `baseline_dev.json` la accuracy en **Train** (42,97 %) y la perplejidad en Dev (8,05).
- [x] Medir y registrar **parámetros entrenables y tiempo de entrenamiento** del baseline (19.908 parámetros, 1,69 s).

---

## Etapa 2 — Red neuronal multicapa y análisis de errores

**Hilo conceptual que tiene que contar esta etapa (es lo que se defiende oralmente):**
diagnóstico del baseline → acción → nuevo diagnóstico → nueva acción.
1. Baseline: error alto en Train y brecha chica con Dev ⇒ **sesgo alto**.
2. Acción: modelo más expresivo (MLP).
3. MLP sin regularizar: brecha Train–Dev grande ⇒ **varianza alta**.
4. Acción: regularización.

### 4.2.1 Análisis de errores manual
- [x] **Diagnóstico cuantitativo primero:** error de Train vs Dev del baseline (57,03 % vs 59,17 %).
- [x] Estimar una **cota inferior del sesgo evitable** (sección 8.5): el ganador de la Etapa 2 baja el error de Train a 30,57 %, 26,46 pp menos que el baseline.
- [x] **Data mismatch:** evidencia de la Etapa 1 (frecuencias estables entre mitades del cuento).
- [x] Tomar **≥ 50 errores de Dev al azar** (semilla fija), con contexto, real y predicho (`muestra_50_errores_dev.csv`).
- [ ] ❗ Etiquetar **a mano** cada error con una **causa**, tal como pide la consigna. Hoy están pre-etiquetados por reglas; falta que el grupo los revise y corrija a mano (la checklist del notebook lo marca pendiente).
- [x] Agregar el **tipo descriptivo** como columna aparte (columna "Categoría Lingüística").
- [x] Tabla de conteos por causa (gráfico de barras de la sección 6). Con 50 muestras el error de estimación es ≈ ±14 pp: se presenta como análisis cualitativo.
- [x] Para cada causa, **qué cambio la corregiría** (sección 6, celda de causas).

### 4.2.2 Arquitectura MLP
- [x] **Mismo input que el baseline** (últimos 5 caracteres one-hot = 320 entradas), para aislar el efecto de la no linealidad.
- [x] **3 capas ocultas:** 128-64-32 con ReLU, salida softmax de 64 clases, entropía cruzada y Adam (53.536 parámetros).
- [x] Documentar en una tabla: capas, neuronas, activaciones, pérdida, optimizador, learning rate, batch size, épocas, parámetros.
- [x] Entrenar **sin regularización** (100 épocas, 5 semillas), con grilla de learning rate (elegido: 0,0005).
- [x] Curvas de **loss y accuracy**, Train vs Dev, marcando la época del mínimo de la pérdida en Dev.
- [x] Nuevo diagnóstico: brecha Train–Dev de 24,17 pp (Train 71,07 %, Dev 46,90 %) ⇒ varianza alta.
- [x] Registrar parámetros, tiempo, accuracy y perplejidad en Dev.

### 4.2.3 Regularización
- [x] Aplicar **≥ 2 técnicas**: Dropout, L2, Early Stopping y Batch Normalization.
- [x] **Ablación:** cada técnica sola y combinación incremental con umbral = desvío entre semillas (5 semillas cada una).
- [x] Hiperparámetros con una **grilla chica en Dev**: Dropout {0,05; 0,1; 0,2; 0,3}, L2 {1e-6; 1e-5; 1e-4}, paciencia {5; 10; 20}.
- [x] Curvas **superpuestas** sin vs con regularización (loss y accuracy).
- [x] Reportar la **mejora vs baseline**: ganador Dropout 0,05 = 48,71 % ± 0,52 % (**+7,88 pp**); brecha de 24,17 a 20,72 pp.
- [x] Aclarar que Early Stopping usa Dev para decidir cuándo parar, por lo que el número en Dev es levemente optimista.
- [x] Documentar lo que no funcionó (sección 8.7).
- [ ] L2 quedó en el borde del rango (1e-4): probar un valor mayor queda como experimento pendiente (opcional).

**Cierre de Etapa 2**
- [x] Tabla comparativa en Dev: baseline vs MLP sin regularizar vs MLP regularizado (accuracy ± desvío, perplejidad, brecha, parámetros, tiempo)
- [x] Checklist formal de la consigna 4.2.x
- [x] Interpretaciones y conclusiones (secciones 8.1, 8.2, 8.6)
- [ ] Commit

---

## Etapa 3 — Arquitectura del dominio (texto)

- [x] **Embeddings (4.3.B):**
  - Embeddings de caracteres desde cero (16 y 300 dim.) contra el one-hot, sección 2.
  - Pre-entrenados vs desde cero con palabras: siguiente palabra con GloVe 6B 100d, sección 8 (no compite con el resto). GloVe no tiene vectores de caracteres.
- [x] Variar el largo del contexto: LSTM con 5 y con 40 caracteres; las dos con el esquema de la clase 8 (Adam con `clipnorm`, Early Stopping) y el lr de la Etapa 2 (R3 y R4).
- [x] Red densa sobre embeddings y modelo de lenguaje simple (LSTM), arquitectura documentada.
- [x] Documentar **≥ 10 ejemplos concretos** que siguen siendo difíciles (sección 5), más accuracy por tipo y por posición en la palabra.
- [x] Reportar la **mejora vs Etapa 2** con 5 semillas: R4 — LSTM de 5 caracteres con Early Stopping = 50,07 % ± 0,54 % (**+1,36 pp**).
- [x] Caché de R2 (`data/processed/cache_R2.json`) para no reentrenarlo en cada corrida.
- [x] Registrar parámetros, tiempo y métricas en la tabla de resultados.
- [x] Interpretaciones, conclusiones y experimentos fallidos (secciones 3, 5, 6, 7, 7.1 y 8), reescritas con la corrida del 21/11.
- [x] Revisión manual de los textos.

---

## Etapa 4 — Modelo generativo (OPTATIVA)

- [x] Decidido: no se hace; queda como trabajo futuro en las conclusiones de la Etapa 5 (por ejemplo, un VAE sobre ventanas de caracteres).

---

## Etapa 5 — Cierre, comparación y ética

Notebook: `notebooks/ETAPA_5.ipynb` (estructura tomada de la rama `dev_thiago`, adaptada a nuestra partición).

- [x] **4.5.1** Tabla comparativa de todos los modelos: arquitectura, parámetros entrenables, Dev, Test (solo el ganador) y tiempo. Gráfico y discusión de complejidad vs performance.
- [x] **4.5.2** El ganador (R4) se reentrena con sus 5 semillas, reproduce su Dev (50,07 %) y se evalúa en Test **una única vez**: **45,70 % ± 0,38 %** (−4,37 pp respecto de Dev). Discusión de overfitting al Dev y de la variación entre bloques.
- [x] Mejoras posibles con más tiempo o recursos.
- [x] **4.5.3** Sesgos medibles en el texto (género, vocabulario), desempeño por subgrupos en Test (tipo de carácter, frecuencia, bloque), riesgos y mitigación.
- [x] **4.5.4** Conclusiones: aprendizajes, lo más desafiante y cómo se resolvió, limitaciones.
- [ ] La consigna 4.5.1 pide "métrica en Test" por modelo y la 4.5.2 evaluar en Test solo el mejor: se tomó la lectura estricta (solo el ganador). Consultar en el Foro si hace falta.
- [x] Revisión manual de los textos.

---

## Entrega

- [x] Notebook **único**, con un separador por etapa (`entrega/.../TP_IA2026_CARABAJAL_FERNANDEZ_MONTENEGRO_GRUPO9.ipynb`)
- [x] Nombre del archivo: `TP_IA2026_CARABAJAL_FERNANDEZ_MONTENEGRO_GRUPO9.ipynb`
- [x] Estructura del `.zip`: `TP_IA2026_CARABAJAL_FERNANDEZ_MONTENEGRO_TheFestival/` con notebook, `requirements.txt`, `README.md`, `resultados/`, más `db/`, `data/processed/` y `experimentos/` (historial y experimentos retirados)
- [x] `requirements.txt` con versiones
- [x] `README.md`: cómo reproducir el entorno, dónde ubicar `The_Festival.txt` y en qué orden ejecutar
- [x] Rutas relativas: el notebook usa la carpeta base de la Etapa 1 (`BASE_DIR`)
- [x] Prueba de principio a fin del notebook integrado (copia con 1 época, 2 semillas y Test reemplazado por Dev): 55 celdas sin errores, 6,7 min. La entrega conserva las salidas de las corridas completas del 21/11, para no evaluar Test dos veces.
- [ ] Una sola entrega por grupo, a cargo de un integrante

---

## Calendario sugerido (ajustalo a las fechas del Campus)

| Fechas | Objetivo |
|---|---|
| 02/11 | Mandar la consulta del docente sobre 4.3.B; cerrar pendientes de Etapa 1 |
| 02/11 – 06/11 | Etapa 2 completa |
| 07/11 – 14/11 | Etapa 3 |
| 15/11 – 18/11 | Etapa 5 (incluye la única evaluación en Test) |
| 19/11 – 20/11 | Integración, README, requirements, ejecución limpia de principio a fin (colchón) |
| 21/11 | Entrega |
| Solo si sobra tiempo | Etapa 4 |

---

## Tabla de resultados (se completa a medida que avanzás; alimenta la Etapa 5)

| Modelo | Parámetros entrenables | Tiempo de entrenamiento | Accuracy Dev (media ± desvío) | Perplejidad Dev | Brecha Train–Dev | Notas |
|---|---|---|---|---|---|---|
| Mayoritaria | — | — | 17,85 % | ≈ 19,3 | — | referencia trivial |
| Baseline (Reg. Logística, K = 5) | 19.908 | 1,92 s | 40,83 % | 8,05 | 2,13 pp | Etapa 1 |
| MLP sin regularizar (128-64-32, lr 0,0005) | 53.536 | 25,0 s por corrida | 46,90 % ± 0,49 % | 11,11 | 24,17 pp | Etapa 2 |
| MLP + Dropout 0,05 | 53.536 | 28,4 s por corrida | 48,71 % ± 0,52 % | 7,66 | 20,72 pp | ganador Etapa 2 |
| E1 — Embedding desde cero 16 dim. + red densa | 23.840 | 25 s por corrida | 47,88 % ± 0,73 % | 6,51 | 10,11 pp | Etapa 3 |
| E2 — Embedding desde cero 300 dim. + red densa | 223.776 | 40 s por corrida | 49,51 % ± 0,81 % | 8,66 | 25,62 pp | mejor embedding de caracteres |
| R1 — LSTM, 5 caracteres | 247.104 | 100 s por corrida | 49,69 % ± 0,12 % | 7,64 | 28,67 pp | Etapa 3 |
| R2 — LSTM, 40 caracteres | 247.104 | 583 s por corrida | 45,06 % ± 1,70 % | 8,54 | 27,41 pp | sobreajusta |
| R3 — LSTM 40 + ES | 247.104 | 284 s por corrida | 47,76 % ± 0,55 % | 6,16 | 9,20 pp | mejor LSTM de 40 |
| R4 — LSTM 5 + ES | 247.104 | 45 s por corrida | 50,07 % ± 0,54 % | 5,91 | 10,43 pp | **ganador Etapa 3; Test: 45,70 % ± 0,38 %** |

Tiempos medidos en CPU (Windows, corridas locales del 21/11).

## Log de experimentos (incluí los que no funcionaron)

Cifras de las corridas locales (5 semillas, Dev). Las filas del 13–14/11 que ya no están en el notebook se detallan en `docs/EXPERIMENTOS_RETIRADOS.md`.

| Fecha | Qué se probó | Resultado en Dev | Conclusión |
|---|---|---|---|
| 13/11 | Learning rate {1e-3; 5e-4; 1e-4; 5e-5}, MLP sin regularizar | picos de 47,26 / 47,56 / 44,88 / 41,76 % | se elige 5e-4: mismo pico que 1e-3, pero se degrada más lento |
| 13/11 | ReduceLROnPlateau suave y agresivo | picos de 47,38 y 47,12 % | alinea las curvas, no sube el techo; con Dropout rinde menos que lr fijo (fallido, retirado) |
| 13/11 | Dropout {0,05; 0,1; 0,2; 0,3} | 48,71 / 48,61 / 48,14 / 46,90 % | Dropout 0,05 es la única técnica que mejora más que el desvío |
| 13/11 | L2 {1e-6; 1e-5; 1e-4} | 46,66 – 47,08 % | sin efecto (fallido) |
| 13/11 | Early Stopping (paciencia 5, 10 y 20) | 45,75 % | corta antes del pico de accuracy (fallido) |
| 13/11 | Batch Normalization | 43,10 %, perplejidad 252 | sobreconfianza (fallido) |
| 13/11 | Dropout 0,05 + {L2; ES; BatchNorm} | 48,70 / 47,00 / 48,32 % | ninguna combinación mejora (fallido) |
| 14/11 | Embeddings de caracteres E1–E4 | 47,88 / 49,51 / 38,29 / 49,06 % | desde cero ≥ pre-entrenado; congelado falla por sesgo (E3 y E4 retirados) |
| 14/11 | LSTM de 5 (R1) y de 40 caracteres (R2) | 49,69 % y 45,06 % | R1 gana; R2 sobreajusta |
| 14/11 | R3 = R2 + Plateau + ES; R3 + Dropout 0,2; R3 + Embedding 16 | 48,81 / 49,06 / 45,94 % | corrige R2, pero no supera a R1 (retirados) |
| 21/11 | R3 y R4 con el esquema de la clase 8 (lr 0,005, `clipnorm` 1,0, ES paciencia 5) | 48,40 / 49,20 % | ES corta temprano con ese lr (retirado) |
| 21/11 | R4 con lr 0,0005 y paciencia 5 o 10 | 50,07 % con las dos | la paciencia no cambia nada; el lr sí |
| 21/11 | R3 y R4 con lr 0,0005, `clipnorm` 1,0, ES paciencia 5 | 47,76 / 50,07 % | R4 gana la etapa; el contexto corto gana con los dos lr |
| 21/11 | Siguiente palabra: P0 desde cero / P1 GloVe congelado / P2 GloVe ajustable | top-1 8,08 / 10,81 / 11,72 % | GloVe ayuda en promedio, con mucho desvío; P0 queda en el nivel de "the" en 4 de 5 semillas |
| 21/11 | Etapa 5: R4 en Test (única evaluación) | Test 45,70 % ± 0,38 % | −4,37 pp respecto de Dev |
| 21/11 | R4 con *The Call of Cthulhu* sumado a Train (3 semillas, solo Dev; versión multi-cuento de Brian) | 50,31 → 54,40 % | +4,09 pp: el límite son los datos |
