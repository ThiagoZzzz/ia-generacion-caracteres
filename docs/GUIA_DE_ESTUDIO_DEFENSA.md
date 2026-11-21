# Guía de estudio para la defensa — Etapas 1, 2, 3 y 5

> Material interno del grupo, no forma parte de la entrega. Explica **todo el código** de `notebooks/ETAPA_1_Y_2_CORREGIDA_EN_CLASE.ipynb`, `notebooks/ETAPA_3.ipynb` y `notebooks/ETAPA_5.ipynb`, celda por celda, junto con los conceptos que hay detrás y las preguntas que nos pueden hacer.
>
> **Cómo usarla:** leela con el notebook abierto al lado. Cada sección de esta guía tiene el mismo número que la celda de código del notebook. Las cifras de resultados están al final (sección R), tomadas de la corrida local.

---

## Índice

- [Parte 0. El problema en una página](#parte-0-el-problema-en-una-página)
- [Parte 1. Notebook de las Etapas 1 y 2](#parte-1-notebook-de-las-etapas-1-y-2)
  - [Etapa 1: EDA, partición y baseline](#etapa-1-eda-partición-y-baseline)
  - [Etapa 2: análisis de errores, MLP y regularización](#etapa-2-análisis-de-errores-mlp-y-regularización)
- [Parte 2. Notebook de la Etapa 3](#parte-2-notebook-de-la-etapa-3)
- [Parte 2b. Notebook de la Etapa 5](#parte-2b-notebook-de-la-etapa-5)
- [Parte 3. Conceptos transversales](#parte-3-conceptos-transversales)
- [Parte 4. Preguntas probables de la defensa](#parte-4-preguntas-probables-de-la-defensa)
- [Parte 5. Enfoque alternativo de Thiago (rama `dev_thiago`)](#parte-5-enfoque-alternativo-de-thiago-rama-dev_thiago)
- [Parte 6. Más datos: la versión multi-cuento de Brian (rama `dev_brian`)](#parte-6-más-datos-la-versión-multi-cuento-de-brian-rama-dev_brian)
- [Sección R. Resultados para tener en la cabeza](#sección-r-resultados-para-tener-en-la-cabeza)
- [Glosario rápido](#glosario-rápido)

---

## Parte 0. El problema en una página

**Qué queremos:** dado un fragmento de texto de *The Festival* (Lovecraft), predecir **el carácter siguiente**.

**Cómo lo convertimos en aprendizaje supervisado:**

```
texto:  "...the old town of Kingsport..."
X (contexto, 40 caracteres) ──►  y (carácter siguiente)
"he old town of Kingsport, and the stran"  ──►  "g"
```

- Cada **muestra** es una ventana de $L = 40$ caracteres y su **etiqueta** es el carácter que viene después.
- Es una **clasificación multiclase** con $|V| = 64$ clases (todos los caracteres distintos del texto limpio).
- Las ventanas se generan corriendo de a 1 carácter (paso o *stride* 1). Por eso dos ventanas vecinas comparten 39 de 40 caracteres, y eso condiciona toda la partición de datos.

**La línea argumental del TP** (lo que hay que poder contar de memoria):

1. **Etapa 1.** Un baseline lineal (regresión logística sobre los últimos 5 caracteres) llega a ≈ 40,8 % en Dev, contra 17,9 % de predecir siempre el espacio.
2. **Diagnóstico.** El baseline tiene error alto en Train y en Dev, con una brecha chica entre los dos: es **sesgo alto (underfitting)**.
3. **Etapa 2, acción.** Un MLP, que es más expresivo. Aparece **varianza alta (overfitting)**: la brecha Train–Dev pasa a más de 20 pp.
4. **Etapa 2, acción.** Regularización (Dropout, L2, Early Stopping, Batch Normalization), probando cada técnica sola y después combinándolas. Se gana ≈ 8 pp sobre el baseline.
5. **Etapa 3.** Cambiar la representación (embeddings) y el modelo (LSTM). Mejora poco, +1,36 pp: modelos muy distintos chocan con el mismo techo de ≈ 49–50 %. El ganador es una LSTM de 5 caracteres cortada a tiempo con Early Stopping. **El límite está en los datos** (≈ 20.000 caracteres), no en la arquitectura.
6. **Etapa 5.** El ganador se evalúa una sola vez en Test: 45,70 %, 4,37 pp menos que en Dev. Análisis de sesgos del texto y de desempeño por subgrupos.

---

## Parte 1. Notebook de las Etapas 1 y 2

### Etapa 1: EDA, partición y baseline

#### Celda 0 — Configuración e imports

```python
import os, re, json, time, warnings
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
SEED = 42
np.random.seed(SEED)
```

**Qué hace:**
- `re`: expresiones regulares, para limpiar el texto y contar palabras.
- `Counter`: un diccionario que cuenta ocurrencias (`Counter("aab")` → `{'a': 2, 'b': 1}`).
- `Path` (pathlib): rutas de archivos que funcionan igual en Windows y Linux.
- De scikit-learn se usan cuatro cosas: el codificador one-hot, la regresión logística, el `Pipeline` que encadena las dos, y las métricas.
- `SEED = 42` fija la semilla de numpy, así todo lo aleatorio se puede reproducir.

**Por qué importa la semilla:** hay aleatoriedad en la asignación de bloques, en el muestreo de los 50 errores y en la inicialización de las redes. Con la semilla fija, cualquiera que corra el notebook obtiene la misma partición.

---

#### Celda 1 — Carga del archivo original

```python
if (Path.cwd() / "db").exists(): BASE_DIR = Path.cwd()
elif (Path.cwd().parent / "db").exists(): BASE_DIR = Path.cwd().parent
...
RAW_PATH = next((p for p in candidatos if p.exists()), None)
with open(RAW_PATH, "r", encoding="utf-8", newline="") as f:
    raw = f.read()
```

**Qué hace:**
- Busca la carpeta del proyecto (`BASE_DIR`) probando varias ubicaciones posibles. Así el notebook corre igual desde la carpeta `notebooks/`, desde la raíz o desde Colab.
- `next(generador, None)` devuelve el **primer** candidato que existe, o `None` si no hay ninguno.
- `newline=""` lee el archivo **sin traducir los saltos de línea**. Sin eso, Python convertiría `\r\n` en `\n` al leer, y no podríamos contar cuántos `CRLF` trae el original.
- Imprime: total de caracteres, cantidad de `CRLF`, cantidad de etiquetas `<i>` y los símbolos no alfabéticos. Es el "antes" de la limpieza.

---

#### Celda 2 — Limpieza paso a paso

```python
pasos = [
    ("1. CRLF -> LF", lambda t: t.replace("\r\n", "\n").replace("\r", "\n")),
    ("2. Quitar <i> y </i>", lambda t: re.sub(r"</?i>", "", t)),
    ("3. Comillas curvas a rectas", lambda t: t.replace("’", "'")...),
    ("4. Estandarizar espacios y saltos", lambda t: re.sub(r"\s+", " ", t).strip()),
]
for nombre, fn in pasos:
    text = fn(text)
    registro.append((nombre, len(text), len(set(text))))
```

**Qué hace:** aplica cuatro transformaciones en orden y, después de cada una, registra el largo del texto y el tamaño del vocabulario (`len(set(text))` = cantidad de caracteres distintos).

**Las regex, explicadas:**
- `r"</?i>"`: `<`, después una `/` **opcional** (`?`), después `i>`. Encuentra tanto `<i>` como `</i>`.
- `r"\s+"`: uno o más caracteres de espacio en blanco (espacio, tab, salto de línea). Todo eso se reemplaza por un espacio único.

**Por qué cada paso:**
1. `CRLF` es el fin de línea de Windows (`\r\n`). Si no lo unificamos, `\r` sería una clase más sin significado.
2. `<i>` es maquetación HTML (cursiva), no parte del cuento.
3. Las comillas curvas y rectas significan lo mismo. Unificarlas evita partir una clase en dos.
4. El archivo original corta las líneas a ≤ 69 caracteres por formato. Esos saltos no son del texto: si los dejáramos, el modelo tendría que adivinar dónde cortó la línea el editor.

**Pregunta posible:** *¿Por qué no pasaron todo a minúsculas?* Porque las mayúsculas sí son parte de la tarea (después de un punto viene una mayúscula). Sacarlas haría la tarea más fácil, pero menos real. Conservamos el vocabulario completo (decisión 3 del notebook).

---

#### Celda 3 — Estadísticas básicas del corpus

```python
vocabulary = sorted(set(text))
char_counts = Counter(text)
words = re.findall(r"[A-Za-z]+(?:['-][A-Za-z]+)*", text)
sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
```

**Qué hace:**
- `vocabulary`: la lista ordenada de los 64 caracteres distintos. Ordenarla hace que el índice de cada carácter sea siempre el mismo.
- La regex de palabras, `[A-Za-z]+(?:['-][A-Za-z]+)*`: una o más letras, seguidas opcionalmente de grupos "apóstrofo o guion + letras". Así `old-world` o `hill's` cuentan como una palabra. `(?:...)` es un grupo que **no captura**: agrupa sin devolver el pedazo por separado.
- Las oraciones se estiman cortando en `.`, `!` o `?`. Es una aproximación.
- Imprime un resumen: caracteres, |V|, palabras, palabras distintas, oraciones y nulos.

**"Valores nulos: 0":** en texto no hay celdas vacías como en una tabla. Cada posición tiene un carácter, así que la consigna sobre nulos se responde con "no aplica / 0".

---

#### Celda 4 — Gráfico 1: frecuencia de caracteres

- `vis(c)` convierte el espacio en `<SP>` y el salto de línea en `<NL>`, para que se vean en los ejes.
- `serie = pd.Series(char_counts).sort_values(ascending=False)`: las frecuencias ordenadas de mayor a menor.
- Dos paneles: los **20 más frecuentes** y los **20 menos frecuentes** (la "cola larga").

**Qué muestra:** el **desbalance de clases**. El espacio solo es ≈ 18 % del texto, mientras que hay caracteres que aparecen una vez. Esto justifica dos cosas: comparar siempre contra "predecir el más frecuente" y analizar la accuracy por tipo de carácter.

---

#### Celda 5 — Gráfico 2: longitud de palabras

- `sns.histplot(..., discrete=True)`: histograma con una barra por cada largo entero.
- Dos líneas verticales: la media (punteada) y la mediana.
- Imprime las palabras de 14 caracteres o más.

**Por qué es relevante:** el baseline mira solo los últimos 5 caracteres. Si las palabras son más largas que eso, el modelo ve solo un pedazo de la palabra que tiene que completar. Es un argumento para probar contextos más largos (la LSTM de 40 en la Etapa 3).

---

#### Celda 6 — Gráfico 3: signos de puntuación

Cuenta los caracteres que están en el conjunto `signos` y los grafica con su número arriba de cada barra (`plt.text`). Muestra que la puntuación es parte de lo que hay que predecir, y que algunos signos son muy raros.

---

#### Celda 7 — Diagnóstico: desbalance, outliers y data mismatch

```python
mitad = len(text) // 2
c1, c2 = Counter(text[:mitad]), Counter(text[mitad:])
"proporción": [c1[c] / mitad for c in top15] + [c2[c] / (len(text) - mitad) for c in top15]
raros = [c for c, n in char_counts.items() if n < 10]
```

**Qué hace:**
1. Parte el texto en dos mitades y compara la **proporción** de los 15 caracteres más frecuentes en cada una. Se usa proporción y no conteo porque las mitades pueden diferir en un carácter.
2. Lista las **clases raras**: caracteres con menos de 10 apariciones.
3. Imprime el diagnóstico.

**Conceptos:**
- **Data mismatch** (Ng): Train y Dev/Test vienen de distribuciones distintas. Si el cuento cambiara de estilo a la mitad, un modelo entrenado con el principio fallaría con el final. Las barras de las dos mitades son casi iguales, así que no hay evidencia de mismatch.
- **Outliers en clasificación:** acá no son valores extremos de una variable numérica, sino **clases con muy poco soporte**. El modelo casi no puede aprenderlas.

---

#### Celda 8 — Partición por bloques contiguos (la decisión más importante)

```python
L = 40; BLOQUE = 400
limites = [(i, min(i + BLOQUE, len(text))) for i in range(0, len(text), BLOQUE)]
if limites[-1][1] - limites[-1][0] < BLOQUE // 2:      # último bloque muy corto
    ini_prev = limites[-2][0]
    limites = limites[:-2] + [(ini_prev, len(text))]   # se fusiona con el anterior
orden = rng.permutation(n_bloques)
n_tr = round(0.70 * n_bloques); n_dev = round(0.15 * n_bloques)
bloques_split = {"train": sorted(orden[:n_tr]), "dev": ..., "test": ...}

def ventanas_de_bloques(ids):
    for b in ids:
        ini, fin = limites[b]
        for i in range(ini, fin - L):
            X.append(text[i:i+L]); y.append(text[i+L])
```

**Paso a paso:**
1. Corta el texto en bloques de 400 caracteres. Si el último queda con menos de 200, lo pega al anterior para no tener un bloque minúsculo. Resultan 50 bloques.
2. `rng.permutation(n_bloques)` mezcla los índices de bloque con semilla fija. Los primeros 70 % van a Train (35), el 15 % siguiente a Dev (8) y el resto a Test (7).
3. `ventanas_de_bloques` arma las ventanas **solo dentro de cada bloque**: `range(ini, fin - L)` asegura que la ventana y su etiqueta (`text[i+L]`) quedan dentro del bloque.

**Por qué no `train_test_split` al azar (esto lo preguntan seguro):**
Con stride 1, la ventana $i$ y la $i+1$ comparten 39 caracteres. Si una queda en Train y la otra en Dev, el modelo "ya vio" casi el mismo ejemplo: es **fuga de datos (data leakage)**, y la accuracy en Dev sale inflada. Con bloques, las ventanas de Dev salen de pedazos de texto que el modelo nunca vio.

**Por qué asignar bloques al azar y no "primer 70 % del texto a Train":** un corte temporal dejaría el final del cuento solo en Test. Si el final tuviera otro estilo, mezclaríamos la evaluación con un mismatch. Con bloques al azar, los tres conjuntos cubren todo el relato.

**Costo de la decisión:** se pierden $L = 40$ ventanas por bloque (las que cruzarían el borde). Es el precio de no tener fuga.

**Resultado:** 12.780 / 2.880 / 2.520 muestras (Train / Dev / Test).

---

#### Celda 9 — Verificación de proporciones en Train, Dev y Test

```python
freq = pd.DataFrame({s: pd.Series(c).reindex(vocabulary, fill_value=0) for s, c in conteos.items()})
prop = freq / freq.sum()
df_p = prop.loc[top15].reset_index(names="carácter").melt(...)
sin_train = [... freq.loc[c, "Train"] == 0 and (Dev + Test) > 0]
```

**Qué hace:**
- Construye una tabla con 64 filas (caracteres) y 3 columnas (conjuntos) con el conteo de cada carácter **objetivo**. `reindex(..., fill_value=0)` agrega con 0 los caracteres que no aparecen en un conjunto.
- `freq / freq.sum()` divide cada columna por su total, y así pasa a proporciones.
- `melt` pasa la tabla de formato "ancho" a "largo" (una fila por carácter y conjunto), que es lo que pide seaborn para las barras agrupadas.
- Lista los caracteres que aparecen en Dev o Test pero **no** en Train. Resulta solo el `2`: un dígito de un año.

**Concepto:** la consigna pide verificar que las clases estén proporcionadas. Con bloques no podemos estratificar (no se puede forzar la proporción de cada carácter), así que lo **verificamos a posteriori**.

---

#### Celda 10 — Codificación numérica y persistencia

```python
char2idx = {c: i for i, c in enumerate(vocabulary)}
idx2char = {i: c for c, i in char2idx.items()}
(OUT_DIR / "The_Festival_clean.txt").write_text(text, encoding="utf-8")
json.dump({"L": L, "BLOQUE": BLOQUE, "SEED": SEED, "limites_bloques": limites,
           "bloques": bloques_split, "vocabulario": "".join(vocabulary)}, f)
```

- `char2idx` y `idx2char` son los dos sentidos del mapeo carácter ↔ entero. Las redes trabajan con números.
- Se guardan el texto limpio y la partición en `data/processed/`. **La Etapa 3 los lee de ahí**, así que usa exactamente los mismos datos sin repetir la limpieza.

---

#### Celda 11 — Referencia trivial (clase mayoritaria)

```python
clase_mayoritaria = serie.index[0]          # el espacio
acc_dummy_dev = np.mean([y == clase_mayoritaria for y in y_dev])
```

El clasificador más tonto posible: predecir siempre el espacio. Da **17,85 %** en Dev. Cualquier modelo tiene que superarlo con holgura; si no, no aprendió nada útil.

---

#### Celda 12 — Baseline: regresión logística con K ∈ {1, 3, 5}

```python
def ultimos(X, k):
    return np.array([list(x[-k:]) for x in X])     # (N, k): un carácter por columna

pipe = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ("clf", LogisticRegression(max_iter=500, random_state=SEED)),
])
pipe.fit(ultimos(X_train, k), y_train)
K_OPTIMO = int(df_k["Accuracy Dev"].idxmax())
```

**Qué hace:**
- `ultimos(X, k)` se queda con los últimos `k` caracteres de cada ventana y los pone uno por columna. Por ejemplo, `"...the stran"` con k = 3 queda `['r','a','n']`.
- **`OneHotEncoder`** convierte cada columna categórica en columnas binarias, una por cada carácter visto en esa posición. Con k = 5 quedan ≈ 5 × 63 columnas.
- `handle_unknown="ignore"`: si en Dev aparece un carácter que no estaba en Train en esa posición, todas sus columnas quedan en 0 en lugar de dar error.
- **`LogisticRegression`** multiclase. Con el solver por defecto (`lbfgs`) es **multinomial**: una softmax sobre una combinación lineal de las entradas. Por defecto lleva **regularización L2** con `C = 1.0`.
- `Pipeline` encadena los dos pasos: `fit` ajusta el encoder y después el clasificador, y `predict` aplica los dos en orden.
- Prueba k = 1, 3 y 5, mide la accuracy en Train y en Dev y el tiempo, y elige **K con la mejor accuracy en Dev**. Gana K = 5.

**Concepto clave: regresión logística multinomial = red de una capa.**
$$P(y = c \mid x) = \text{softmax}(Wx + b)_c = \frac{e^{w_c \cdot x + b_c}}{\sum_j e^{w_j \cdot x + b_j}}$$
Es exactamente una capa densa con softmax, sin capas ocultas. Por eso es un buen baseline para un TP de redes: el MLP de la Etapa 2 le "agrega capas ocultas" a este mismo modelo.

**Limitación (explica el sesgo):** el modelo es **lineal en el one-hot**. Cada carácter en cada posición suma un "voto" independiente, y no puede representar **interacciones**, del tipo "si hay `t` en la posición 4 **y** `h` en la 5, entonces `e`". Un MLP sí puede, gracias a las capas ocultas no lineales.

**Por qué K = 5 y no 40:** con one-hot, 40 posiciones darían ≈ 2.500 columnas con apenas 12.780 muestras; además, lo que más informa son los caracteres más cercanos. Lo comprobamos: la accuracy sube de K = 1 a 5.

---

#### Celda 13 — Reporte oficial del baseline

```python
def perdida_log(pipe, X_k, y_real):
    proba = pipe.predict_proba(X_k)
    col = {c: j for j, c in enumerate(pipe.classes_)}
    p_real = np.array([proba[i, col[c]] if c in col else 1e-15 for i, c in enumerate(y_real)])
    return float(-np.mean(np.log(np.clip(p_real, 1e-15, 1.0))))
perp_base_dev = float(np.exp(loss_base_dev))
params_base = int(clf_base.coef_.size + clf_base.intercept_.size)
```

**Qué hace:**
- **Entropía cruzada media:** para cada muestra toma la probabilidad que el modelo le asignó al carácter **real** y promedia $-\log p$. Si el carácter real no está entre las clases del modelo (el `2`, que no aparece en Train), le asigna $10^{-15}$. `np.clip` evita calcular $\log(0)$.
- **Perplejidad** $= e^{\text{loss}}$. Interpretación: "el modelo duda en promedio como si eligiera al azar entre *perplejidad* caracteres igualmente probables". El baseline da ≈ 8,05, es decir, como dudar entre 8 caracteres. Predecir al azar entre los 64 daría 64.
- **Parámetros:** `coef_` (una fila de pesos por clase, una columna por cada entrada one-hot) más `intercept_` (un sesgo por clase). Da 19.908, que es 63 × 316: 63 clases (el `2` no está en Train) por 315 columnas one-hot más el intercepto.
- Guarda todo en `baseline_dev.json`. La Etapa 3 lee de ahí la referencia del baseline.

**Por qué medimos la pérdida además de la accuracy:** la accuracy solo mira si el top-1 acertó. La pérdida mide **cuánta probabilidad** le dio el modelo a la respuesta correcta. Dos modelos con la misma accuracy pueden estar muy distintos de bien calibrados.

---

#### Celda 14 — Matriz de confusión del baseline

```python
cm = confusion_matrix(y_dev, y_pred_dev, labels=labels_all, normalize="true")
ix = [labels_all.index(c) for c in top20]
sns.heatmap(cm[np.ix_(ix, ix)], ...)
errores = [(r, p) for r, p in zip(y_dev, y_pred_dev) if r != p]
```

- `normalize="true"` divide cada **fila** (carácter real) por su total. La diagonal es el *recall* de cada clase: qué porcentaje de las veces que el real era `e` el modelo predijo `e`.
- `np.ix_(ix, ix)` extrae la submatriz de las 20 clases más frecuentes, en filas y columnas a la vez.
- Imprime los pares (real → predicho) más confundidos. Típicamente aparecen letras confundidas con espacio y vocales entre sí.

---

#### Celda 15 — Checklist de la Etapa 1

Solo imprime qué requisito de la consigna se cumple y con qué evidencia. No calcula nada nuevo.

---

### Etapa 2: análisis de errores, MLP y regularización

#### Celda 16 — Configuración de TensorFlow y tensores one-hot

```python
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["KERAS_HOME"] = ...                     # caché de Keras dentro del proyecto
SEEDS = [42, 123, 456, 789, 2026]

def extraer_tensores_onehot(X_raw, y_raw, k, c2i, v_size):
    indices = np.array([[c2i[c] for c in seq[-k:]] for seq in X_raw])           # (N, 5)
    X_oh = keras.utils.to_categorical(indices, num_classes=v_size).reshape(len(X_raw), -1).astype("float32")  # (N, 320)
```

**Qué hace:**
- `TF_CPP_MIN_LOG_LEVEL = 2` oculta los mensajes informativos de TensorFlow.
- **Las 5 semillas:** cada configuración de red se entrena 5 veces, con inicializaciones y órdenes de mini-batch distintos, y se reporta media ± desvío.
- `extraer_tensores_onehot` arma la entrada del MLP en dos pasos:
  1. **Cada carácter pasa a one-hot** con `to_categorical` (clase 6): un vector de 64 con un solo 1 en su casillero. Así ningún carácter "vale más" que otro, cosa que pasaría si los numeráramos `a=0, b=1, …`.
  2. **Los 5 vectores se ponen uno al lado del otro** con `reshape`: cada ejemplo queda como un vector de $5 \times 64 = 320$, y el tramo $i$ dice qué carácter hay en la posición $i$.

  Es la **misma información que recibe el baseline**, y eso permite aislar el efecto de la red.
- Las etiquetas quedan como enteros (`y_idx`), no en one-hot. Por eso después se usa `sparse_categorical_crossentropy`.

**Por qué 5 semillas:** una red entrenada dos veces con semillas distintas da resultados distintos. Si comparamos modelos con una sola corrida, una diferencia de 0,5 pp puede ser pura suerte. Con 5 corridas vemos la media y cuánto varía (el desvío).

---

#### Celda 17 — Análisis de errores por reglas y muestra de 50 casos

```python
ctx_train_stats = defaultdict(list)
for ctx, yr, yp in zip(ctx_train_k, y_train, y_pred_tr_base):
    ctx_train_stats[ctx].append((yr, yp))       # para cada 5-grama de Train: (real, predicho)

def clasificar_causa_teorica(ctx, yr, yp):
    if ctx not in ctx_train_stats:               return "Falta de cobertura (5-grama nuevo)"
    conteo = Counter(t[0] for t in ctx_train_stats[ctx]); moda = conteo.most_common(1)[0][0]
    if len(conteo) > 1 and yr != moda:           return "Ruido irreducible (Error Bayes)"
    if aciertos / len(ocurr) <= 0.5:             return "Sesgo (Underfitting)"
    return "Varianza (Overfitting)"
df_50_muestra = df_errores_all.sample(n=50, random_state=SEED)
```

**Qué hace:**
1. Para cada contexto de 5 caracteres que aparece en Train, guarda qué caracteres lo siguieron y qué predijo el baseline. `defaultdict(list)` crea la lista vacía la primera vez que aparece una clave.
2. Junta **todos** los errores del baseline en Dev.
3. A cada error le asigna una **causa** con reglas fijas, aplicadas en orden (gana la primera que se cumple):
   - **Falta de cobertura:** ese 5-grama nunca apareció en Train.
   - **Ruido irreducible:** en Train, ese contexto fue seguido por varios caracteres distintos y el real no es el más común. Incluso el predictor óptimo, que elige siempre la moda, fallaría ahí.
   - **Sesgo:** el contexto está en Train, pero el baseline también falla ahí la mayoría de las veces. Tiene los datos y no aprendió el patrón.
   - **Varianza:** en Train acierta ese contexto, pero en Dev falla.
4. Le asigna además una **categoría lingüística** (predijo espacio de más, omitió un espacio, confundió vocales, etc.).
5. Toma **50 errores al azar** con semilla fija, los guarda en CSV y grafica las dos clasificaciones.

**Por qué reglas y no "a ojo":** para que la clasificación sea reproducible y no dependa de quién mira. **Ojo:** la consigna dice "análisis de errores **manual**". La checklist del notebook marca como pendiente la revisión manual de esas 50 pre-etiquetas. Hay que hacerla o justificar el criterio en la defensa.

**Concepto: el error irreducible (error de Bayes).** Después de `"the "` puede venir casi cualquier letra. Ningún modelo que mire solo esos 5 caracteres puede acertar siempre. Es el piso de error con ese contexto, y por eso no todo el error es "corregible".

**Por qué "± 14 pp" en el título del gráfico:** con 50 casos, una proporción estimada tiene un margen al 95 % de ≈ $1{,}96\sqrt{0{,}25/50} \approx 0{,}14$. Los conteos por causa son orientativos, no precisos.

---

#### Celda 18 — Accuracy por tipo de carácter

```python
def tipo_caracter(c):  # espacio / mayúscula / vocal / consonante / puntuación-dígito
def accuracy_por_tipo(y_pred, y_real=y_dev, tipos=tipos_dev):
    res = {t: (y_pred[tipos == t] == y_real[tipos == t]).mean() for t in TIPOS_CHAR}
```

Usa **máscaras booleanas** de numpy (`tipos == t`) para calcular la accuracy solo sobre los casos de cada tipo. Muestra que el baseline acierta muy bien los espacios (≈ 86 %) y mal las letras (≈ 31–33 %). Lo difícil es elegir la letra **dentro** de la palabra.

---

#### Celda 19 — Explicación de predicciones individuales

```python
def proba_baseline(contextos):     # reordena las columnas de predict_proba al orden de char2idx
def explicar_prediccion(proba_fn, contexto, y_real, nombre_modelo, top=5):
    orden = np.argsort(p)[::-1]                                  # índices de mayor a menor probabilidad
    puesto_real = int(np.where(orden == char2idx[y_real])[0][0]) + 1
```

- `predict_proba` de sklearn devuelve las columnas en el orden de `classes_`, que puede no coincidir con `char2idx` (por ejemplo, falta el `2`). `proba_baseline` las reubica en una matriz de 64 columnas.
- `explicar_prediccion` muestra el top-5 y **en qué puesto quedó el carácter real**.
- `np.argsort(p)[::-1]`: `argsort` ordena de menor a mayor y `[::-1]` invierte el orden.

**Para qué sirve:** distinguir causas. Si el real quedó 2.º con una probabilidad parecida al 1.º, es ruido o ambigüedad. Si quedó con probabilidad ≈ 0, el modelo no aprendió ese patrón.

---

#### Celda 20 — Constructor del MLP y funciones de entrenamiento

```python
def construir_mlp(k=5, v_size=64, dropout_rate=0.0, l2_reg=0.0, batchnorm=False):
    reg = regularizers.l2(l2_reg) if l2_reg > 0 else None
    capas = [layers.Input(shape=(k * v_size,))]
    for units in [128, 64, 32]:
        if batchnorm:
            capas += [layers.Dense(units, kernel_regularizer=reg), layers.BatchNormalization(), layers.Activation("relu")]
        else:
            capas.append(layers.Dense(units, activation="relu", kernel_regularizer=reg))
        if dropout_rate > 0.0:
            capas.append(layers.Dropout(dropout_rate))
    capas.append(layers.Dense(v_size, activation="softmax", kernel_regularizer=reg))
    return keras.Sequential(capas)
```

**La arquitectura:** entrada de 320 → Dense 128 (ReLU) → Dense 64 (ReLU) → Dense 32 (ReLU) → Dense 64 (softmax).
- **3 capas ocultas** (lo que pide la consigna), con tamaños decrecientes: la red va comprimiendo la información.
- **ReLU** $= \max(0, z)$: la no linealidad que permite representar interacciones. Sin ella, apilar capas densas seguiría siendo un modelo lineal.
- **Softmax** a la salida: convierte 64 números en 64 probabilidades que suman 1.
- Un único constructor sirve para todas las variantes: los parámetros activan Dropout, L2 o BatchNorm. Así todas las configuraciones comparten la misma base.
- Con BatchNorm el orden es **Dense → BatchNorm → ReLU**: se normaliza la entrada de la activación.

**Parámetros (para responder de memoria):** una capa densa de $n_{in}$ a $n_{out}$ tiene $n_{in} \cdot n_{out} + n_{out}$ parámetros (pesos más sesgos).
$41.088 + 8.256 + 2.080 + 2.112 = 53.536$. Con 12.780 muestras son ≈ 4,2 parámetros por muestra: hay capacidad de sobra para memorizar.

```python
def entrenar_mlp(seed, lr, dropout=0.0, l2=0.0, batchnorm=False, use_es=False, patience=0, epochs=EPOCAS):
    keras.utils.set_random_seed(seed)
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                   loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    if use_es:  cbs.append(callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True))
    h = modelo.fit(X_train_oh, y_train_idx, validation_data=(X_dev_oh, y_dev_idx), epochs=epochs, batch_size=BATCH, callbacks=cbs, verbose=0)
```

- `keras.utils.set_random_seed(seed)` fija a la vez las semillas de Python, numpy y TensorFlow.
- **Adam:** optimizador que adapta el paso de cada peso según el historial de sus gradientes. El *learning rate* es el tamaño base del paso.
- **`sparse_categorical_crossentropy`:** la misma entropía cruzada de siempre, pero recibe la etiqueta como entero (`7`) en lugar de one-hot (`[0,…,1,…,0]`).
- **Callbacks:** funciones que Keras llama al final de cada época. `EarlyStopping` corta si `val_loss` deja de mejorar durante `patience` épocas y recupera los mejores pesos.
- `validation_data=(X_dev, y_dev)`: después de cada época evalúa en Dev. **Ojo:** Dev se usa para monitorear y, con Early Stopping, para decidir cuándo cortar. Por eso el número en Dev queda levemente optimista y la medición honesta es la de Test (Etapa 5).
- **Batch 128:** los pesos se actualizan cada 128 muestras, unas 100 actualizaciones por época.

```python
def evaluar_config(cfg):
    for s in SEEDS:
        modelo, h, t_run = entrenar_mlp(s, **cfg)
        ...
        "pico_acc": max(va), "ep_pico_acc": argmax(va)+1,
        "min_loss": min(vl), "ep_min_loss": argmin(vl)+1,
        "acc_en_min_loss": va[argmin(vl)], "distancia": ep_pico_acc - ep_min_loss
    resumen[f"{col}_mean"], resumen[f"{col}_std"] = ...
```

Entrena una configuración con las 5 semillas y guarda, por corrida, el tiempo, la accuracy y la pérdida en Train y en Dev, la brecha y la perplejidad, más cuatro métricas de las **curvas**:
- `pico_acc` y su época: la mejor accuracy en Dev de todo el entrenamiento.
- `min_loss` y su época: el mínimo de la pérdida en Dev.
- `acc_en_min_loss`: la accuracy en la época del mínimo de la pérdida, que es lo que entregaría Early Stopping.
- `distancia`: cuántas épocas hay entre el pico de accuracy y el mínimo de la pérdida.

Después promedia todo (media y desvío). El modelo y el historial de la semilla 42 se guardan para graficar.

---

#### Celda 21 — Ajuste del learning rate

```python
LRS = [1e-3, 5e-4, 1e-4, 5e-5]
for lr in LRS: resultados_lr[lr], ... = evaluar_config({"lr": lr})
LR_OPT = max(LRS, key=lambda lr: resultados_lr[lr]["pico_acc_mean"])
```

**Qué hace:** entrena el MLP **sin regularizar** con 4 learning rates (5 semillas × 100 épocas cada uno), arma una tabla y elige el de **mayor pico medio de accuracy en Dev**. También avisa si el elegido quedó en el borde del rango probado: si fuera así, convendría probar valores más allá.

**El problema que motivó esto (importante para la defensa):** con lr = 0,001, la pérdida en Dev tocaba su mínimo en la época ≈ 14, pero la accuracy seguía subiendo hasta la ≈ 35. Early Stopping mira la pérdida, así que cortaba **antes de tiempo**.

**¿Por qué la pérdida y la accuracy no coinciden?** Pasado cierto punto, el modelo se vuelve **sobreconfiado**: en los ejemplos donde falla asigna probabilidades muy altas al carácter equivocado, y eso dispara la pérdida ($-\log$ de un número chico es enorme). Mientras tanto, sigue mejorando de a poco en el top-1. La accuracy solo mira el argmax; la pérdida castiga la confianza mal puesta.

---

#### Celda 22 — Curvas del MLP sin regularizar

Grafica, para el lr elegido y la semilla 42, la **pérdida** y la **accuracy** en Train (punteado) y en Dev (línea llena), con una línea vertical en el mínimo de la pérdida en Dev y otra en el pico de accuracy. También dibuja el baseline como referencia horizontal.

**Lo que hay que leer:** la pérdida en Train baja sin parar y la de Dev baja, toca un mínimo y vuelve a subir. Esa "U" es la firma del **sobreajuste**. La brecha Train–Dev (> 20 pp) confirma **varianza alta**.

---

#### Celda 23 — Cada técnica de regularización por separado

```python
GRILLA_IND = {
    "Dropout": [{"dropout": d} for d in [0.05, 0.1, 0.2, 0.3]],
    "L2": [{"l2": v} for v in [1e-6, 1e-5, 1e-4]],
    "Early Stopping": [{"use_es": True, "patience": p} for p in [5, 10, 20]],
    "Batch Normalization": [{"batchnorm": True}],
}
def correr(cfg):        # entrena solo si no se entrenó antes (memoización por nombre)
MEJOR_POR_TECNICA[tecnica] = cfgs[i_mejor]
borde = len(cfgs) > 1 and i_mejor in (0, len(cfgs) - 1)
```

**Qué hace:** una **grilla chica** de valores por técnica, 5 semillas cada uno, y se queda con el mejor valor de cada técnica según la accuracy media en Dev. Avisa si el mejor quedó en el borde del rango. `correr` evita reentrenar configuraciones repetidas: guarda los resultados por nombre.

**Las cuatro técnicas, conceptualmente:**

| Técnica | Qué hace | Por qué reduce la varianza |
|---|---|---|
| **Dropout** (p) | En cada paso de entrenamiento apaga al azar una fracción p de las neuronas. En predicción no apaga nada (Keras escala las activaciones durante el entrenamiento por $1/(1-p)$ para compensar). | La red no puede depender de combinaciones puntuales de neuronas. Equivale aproximadamente a promediar muchas subredes. |
| **L2** (λ) | Suma $\lambda \sum w^2$ a la pérdida. | Empuja los pesos hacia valores chicos y da funciones más suaves, que se ajustan menos al ruido. |
| **Early Stopping** (paciencia) | Corta cuando `val_loss` no mejora durante "paciencia" épocas y vuelve a los mejores pesos. | Limita cuánto se aleja el modelo de la inicialización. Tiene un efecto parecido a L2. |
| **Batch Normalization** | Normaliza la entrada de cada activación con la media y el desvío del mini-batch, y después la reescala con dos parámetros aprendidos. En predicción usa medias móviles. | Sobre todo estabiliza el entrenamiento. Regulariza poco, por el ruido de calcular las estadísticas en cada batch. |

**Nota:** con Dropout, la pérdida de **Train** que muestra `fit` se calcula con neuronas apagadas, y por eso puede quedar por encima de la de Dev. El notebook mide la accuracy de Train aparte con `evaluate`, que no aplica Dropout.

---

#### Celda 24 — Combinación incremental (selección hacia adelante)

```python
while restantes:
    candidatos = [(t, {**actual_cfg, **MEJOR_POR_TECNICA[t]}) for t in restantes]
    t_mejor, cfg_mejor, n_mejor = max(candidatos, key=dev_acc_mean)
    umbral = resultados_ablacion[actual]["dev_acc_std"]
    acepta = ganancia > umbral
    if not acepta: break
    actual_cfg, actual = cfg_mejor, n_mejor; restantes.remove(t_mejor)
GANADOR = actual
json.dump(registro_etapa2, ...)   # resultados_modelos.json
```

**Algoritmo:**
1. Arranca del MLP sin regularizar.
2. En cada paso prueba sumarle **cada técnica que falta** (con su mejor valor de la celda 23).
3. Se queda con la que más mejora, **solo si** la mejora supera el desvío entre semillas del modelo actual. Si no, para.
4. El último modelo aceptado es el **ganador de la Etapa 2**.

`{**a, **b}` une dos diccionarios; si una clave se repite, gana la de `b`.

**Por qué el umbral es el desvío:** para no aceptar "mejoras" que son ruido de la semilla. Es un criterio simple y conservador.

**Por qué incremental y no "todo junto":** si combinamos todo y empeora, no sabemos qué técnica tuvo la culpa. Además, algunas técnicas se pisan entre sí (ver "Qué esperamos al combinarlas" en el notebook): Early Stopping y L2 hacen algo parecido, y Dropout y BatchNorm pueden chocar.

**Persistencia:** guarda **todo** en `resultados_modelos.json`: baseline, learning rates, grilla, pasos y ganador. La Etapa 3 lee de ahí la configuración ganadora, y la Etapa 5 la va a usar para la tabla.

---

#### Celda 25 — Tabla de síntesis y curvas con y sin regularización

Arma la tabla comparativa (baseline + MLP base + mejor de cada técnica + combinaciones), con parámetros, tiempo, accuracy en Train y Dev, peor semilla, brecha, pérdida y perplejidad. Grafica:
- **Figura 1:** pérdida y accuracy en Train y Dev del MLP sin regularizar contra el ganador. Es la comparación "con vs sin regularización" que pide la consigna.
- **Figura 2:** curvas de Dev de todas las variantes.

`list(dict.fromkeys(lista))` elimina duplicados **conservando el orden**.

---

#### Celda 26 — ¿Dónde mejora el MLP?

Calcula la accuracy por tipo de carácter del ganador y la compara con la del baseline (columna Δ). Vuelve a explicar los mismos 4 casos de la celda 19, ahora con los dos modelos. Sirve para responder **en qué** mejora la red, más allá del número global.

---

#### Celda 27 — Estimación del sesgo evitable

```python
err_tr_base    = 1 - acc_base_train
err_tr_ganador = 1 - resultados_ablacion[GANADOR]["train_acc_mean"]
print(err_tr_base - err_tr_ganador)    # reducción del error de Train
```

**Concepto (Ng):** el error de Train se descompone en **error irreducible + sesgo evitable**. Si un modelo más flexible baja el error de Train de 57,03 % a 30,57 % (lo que logra el ganador de la Etapa 2), el baseline tenía **al menos** esa diferencia de sesgo evitable.

**Por qué no se usa el MLP sin regularizar:** memoriza Train y su error de Train puede quedar **por debajo** del irreducible. Usarlo como referencia exageraría el sesgo evitable. Por eso el notebook usa el ganador regularizado.

---

#### Celda 28 — Checklist de la Etapa 2

Imprime el estado de cada requisito. La primera línea dice `[ ]`: **falta la revisión manual de las 50 pre-etiquetas**.

---

## Parte 2. Notebook de la Etapa 3

#### Celda 0 — Configuración

Los mismos imports más `urllib.request` (para descargar GloVe en la sección 8) y `PCA` de sklearn (para visualizar los embeddings). Las mismas `SEEDS`.

---

#### Celda 1 — Carga de datos de la Etapa 1

```python
text = (PROC / "The_Festival_clean.txt").read_text(encoding="utf-8")
splits = json.load(open(PROC / "splits_bloques.json"))
a_indices = lambda X: np.array([[char2idx[c] for c in x] for x in X], dtype=np.int32)
Xi_train, Xi_dev = a_indices(X_train), a_indices(X_dev)
```

- **Reutiliza** el texto limpio y la partición guardados en la Etapa 1, y rearma las ventanas con la misma función. Así los datos son idénticos.
- **Cambio importante respecto de la Etapa 2:** la entrada ahora son **índices enteros** de forma $(N, 40)$, no one-hot. La capa `Embedding` de Keras recibe índices y los convierte en vectores por dentro.
- **Test no se carga**: ni siquiera se generan sus ventanas.
- `vis = lambda c: {" ": "<SP>", "\n": "<NL>"}.get(c, c)` muestra el espacio y el salto de línea de forma legible en tablas y gráficos. `.get(c, c)` devuelve el reemplazo si existe, y si no, el mismo carácter.

---

#### Celda 2 — Referencias de las Etapas 1 y 2

Lee `baseline_dev.json` y `resultados_modelos.json` y hereda la configuración del ganador de la Etapa 2: learning rate, Dropout, épocas, batch y K. **Mismas condiciones de entrenamiento**, así la diferencia se debe a la representación o al modelo.

---

#### Celda 3 — Red densa sobre embeddings (el corazón de la 4.3.B)

```python
def construir_modelo(tipo, k=K, emb_dim=16, emb_init=None, emb_trainable=True, dropout=DROPOUT, unidades_lstm=128):
    entrada = layers.Input(shape=(k,), dtype="int32")
    if tipo == "onehot":
        x = layers.Embedding(V, V, embeddings_initializer=keras.initializers.Identity(), trainable=False)(entrada)
    else:
        init = keras.initializers.Constant(emb_init) if emb_init is not None else "uniform"
        x = layers.Embedding(V, emb_dim, embeddings_initializer=init, trainable=emb_trainable, name="embedding")(entrada)
    if tipo in ("onehot", "densa"):
        x = layers.Flatten()(x)
        for u in [128, 64, 32]: x = Dense(u, relu) (+ Dropout)
    else:
        x = layers.LSTM(unidades_lstm)(x) (+ Dropout)
    salida = layers.Dense(V, activation="softmax")(x)
```

**¿Qué es una capa `Embedding`?** Una **tabla** de $|V| \times d$: la fila $i$ es el vector del carácter $i$. Recibe índices y devuelve las filas correspondientes. Matemáticamente es lo mismo que multiplicar el one-hot por una matriz de pesos, pero sin construir el one-hot.

**El truco del modelo "onehot" (E0):** un `Embedding` de $64 \times 64$ inicializado con la **matriz identidad** y congelado devuelve, para el índice $i$, el vector $e_i$, que es exactamente el one-hot. Así E0 reproduce el MLP de la Etapa 2 con el mismo código que los demás.

**`Flatten`:** convierte la salida de forma $(k, d)$, un vector por carácter, en un solo vector de $k \cdot d$ que entra a la red densa. Con $k = 5$ y $d = 16$ son 80 entradas.

**Las tres variantes:**

| Modelo | Embedding | Parámetros | Qué compara |
|---|---|---|---|
| E0 | one-hot (identidad congelada) | 53.536 | Repite el ganador de la Etapa 2 |
| E1 | desde cero, 16 dim. | 23.840 | Efecto de una representación chica |
| E2 | desde cero, 300 dim. | 223.776 | Efecto de una representación grande |

Cuentas: E1 = $64 \cdot 16$ (tabla) $+ (80 \cdot 128 + 128) + 8.256 + 2.080 + 2.112$.

**`"uniform"`:** inicializa la tabla con valores uniformes chicos (en Keras, entre −0,05 y 0,05).

**`evaluar(...)`:** el mismo esquema que `evaluar_config` de la Etapa 2 (5 semillas, mismas métricas), con dos diferencias: recibe un constructor (una `lambda` que arma el modelo) y recorta la entrada a los últimos `k` caracteres (`Xi_train[:, -k:]`). Así el mismo código sirve para K = 5 y para L = 40. Sus parámetros opcionales (`lr`, `clipnorm`, `use_es`, `patience`) se usan en R3 y R4.

**Parámetros entrenables:** `sum(np.prod(w.shape) for w in m42.trainable_weights)` cuenta solo los pesos entrenables. En E0 la tabla identidad está congelada y no cuenta.

**`tabla(...)`:** arma la tabla con la columna **Δ vs Etapa 2**, que es la mejora que pide reportar la consigna.

---

#### Celda 4 — Curvas de validación

`graficar_curvas` dibuja la pérdida y la accuracy en Dev por época (semilla 42) de los modelos pedidos, con una línea horizontal en el ganador de la Etapa 2.

---

#### Celda 5 — LSTM (modelo de lenguaje simple)

```python
MEJOR_EMB = max(candidatos_emb, key=lambda n: resultados[n]["dev_acc_mean"])
MODELOS_LSTM = {
    "R1 — LSTM, 5 caracteres":  (lambda: construir_modelo("lstm", k=K, **cfg_emb), K),
    "R2 — LSTM, 40 caracteres": (lambda: construir_modelo("lstm", k=L, **cfg_emb), L),
}
```

**Qué hace:** toma la **mejor representación** de la celda 3 y entrena dos LSTM: una con los mismos 5 caracteres y otra con la ventana completa de 40.

**Caché de R2:**

```python
CACHE_R2 = PROC / "cache_R2.json"
cfg_r2 = {"emb": cfg_emb, "lr": LR, "dropout": DROPOUT, "epocas": EPOCAS, "batch": BATCH, "seeds": SEEDS}
if nombre == R2 and cache is not None and cache["config"] == cfg_r2:
    resultados[nombre], historiales[nombre] = cache["resultado"], cache["historial"]
```

R2 tarda unos 50 minutos con las 5 semillas. La primera corrida guarda sus resultados y sus curvas en un JSON; las siguientes los cargan en lugar de reentrenar. Si cambia algo de su configuración, el caché no coincide y se vuelve a entrenar. Como las semillas son fijas, cargar o reentrenar da los mismos números. No se guarda el modelo porque las secciones 5 y 6 solo usan el ganador y E2.

**¿Qué es una LSTM?** Una red **recurrente**: lee la secuencia carácter por carácter y mantiene un **estado** (una memoria de 128 números) que se actualiza en cada paso. Tiene tres **compuertas** (de entrada, de olvido y de salida) que deciden qué guardar, qué borrar y qué mostrar. Esas compuertas resuelven el problema de los gradientes que se desvanecen en las RNN simples, y por eso puede usar contextos largos.

**Diferencia con la red densa:** la densa ve los 5 caracteres como un bloque fijo, y cada posición tiene sus propios pesos. La LSTM **comparte los mismos pesos en cada paso** y procesa en orden. Por eso la cantidad de parámetros **no depende del largo del contexto**: R1 y R2 tienen los mismos 247.104.

**Parámetros de una LSTM:** $4 \cdot [u \cdot (d + u) + u]$, con $u$ unidades y $d$ la dimensión de entrada (el 4 sale de las 3 compuertas más la celda candidata). Con $u = 128$ y $d = 300$: $4 \cdot (128 \cdot 428 + 128) = 219.648$. A eso se suman la tabla ($64 \cdot 300 = 19.200$) y la salida ($128 \cdot 64 + 64 = 8.256$), y da 247.104.

**R1 contra R2:** R1 separa el efecto de "leer en orden"; R2 mide si sirve "ver más contexto".

---

#### Celda 6 — La LSTM de 40 caracteres con Early Stopping (R3)

```python
LR_LSTM = LR  # el learning rate elegido en la Etapa 2
CLIPNORM = 1.0
resultados[R3], historiales[R3], modelos[R3] = evaluar(
    lambda: construir_modelo("lstm", k=L, **cfg_emb), k=L, lr=LR_LSTM, clipnorm=CLIPNORM, use_es=True, patience=5)
```

**Por qué hizo falta:** R2 sobreajusta. Llega a su pico alrededor de la época 51 y después cae, porque la configuración heredada (Dropout 0,05, lr fijo, sin Early Stopping) se ajustó para el MLP, no para una LSTM.

**Qué hace:** entrena la misma LSTM de 40 caracteres con el **esquema de la LSTM de la clase 8**:
- **Adam con `clipnorm=1.0`** (*gradient clipping*): si la norma del gradiente supera 1, se recorta a 1. En una red recurrente el gradiente se multiplica muchas veces a lo largo de los pasos y puede "explotar"; el recorte lo evita.
- **Early Stopping** con paciencia 5 sobre `val_loss`, que recupera los pesos de la mejor época.
- **Learning rate:** el elegido en la Etapa 2 (0,0005). El de la clase 8 era 0,005; con él Early Stopping cortaba muy temprano. Está en `docs/EXPERIMENTOS_RETIRADOS.md`.

---

#### Celda 7 — Control: la LSTM de 5 caracteres con el mismo esquema (R4)

```python
resultados[R4], historiales[R4], modelos[R4] = evaluar(
    lambda: construir_modelo("lstm", k=K, **cfg_emb), k=K, lr=LR_LSTM, clipnorm=CLIPNORM, use_es=True, patience=5)
comparadas_33 = ["R1 — LSTM, 5 caracteres", R4, R3]
```

**Por qué hace falta:** R1 y R3 no se entrenaron igual (R1 sin Early Stopping, R3 con el esquema de la clase 8). Comparándolas así no se sabe si la diferencia viene del contexto o del entrenamiento. R4 es R1 con el mismo esquema que R3: ahora R3 y R4 solo difieren en el largo del contexto. Es un **experimento de control**: se cambia una sola cosa por vez.

**Resultado clave:** R4 resulta ser **el ganador de la etapa**. Es R1 cortado en el mínimo de la pérdida: misma accuracy en la práctica, pero mucho menos sobreajuste, mejor perplejidad y la mitad del tiempo.

---

#### Celda 8 — Comparación con la Etapa 2

```python
GANADOR3 = max(resultados, key=lambda n: resultados[n]["dev_acc_mean"])
print(mejora sobre la Etapa 2 y sobre el baseline)
json.dump({...}, open(PROC / "resultados_etapa3.json", "w"))
```

Elige el ganador de la Etapa 3 por **accuracy media en Dev**, reporta la mejora (Δ en pp) sobre la Etapa 2 y sobre el baseline, y guarda todo en `resultados_etapa3.json` para la Etapa 5.

---

#### Celda 9 — Accuracy por tipo de carácter y por posición en la palabra

```python
def posicion_en_palabra(ctx):
    p = len(ctx) - 1 - ctx.rfind(" ")      # caracteres de la palabra actual ya vistos
    return str(p) if p < 5 else "5+"
aciertos = {n: proba(n).argmax(axis=1) == yi_dev for n in comparados}
```

- `ctx.rfind(" ")` devuelve la posición del **último** espacio del contexto. Si el contexto termina en espacio, `p = 0`: hay que adivinar la primera letra de una palabra nueva.
- Compara E0 (Etapa 2) contra el ganador de la Etapa 3, por tipo de carácter y por posición.

**Qué esperar:** la **posición 0** (la primera letra de la palabra) es la más difícil, porque es la que más depende del significado y no de la ortografía. A medida que avanza la palabra, la ortografía restringe las opciones y la accuracy sube.

---

#### Celda 10 — Diez errores concretos del ganador

```python
errores = np.where(pred_gan != yi_dev)[0]
for t in TIPOS_CHAR:
    idx_t = errores[tipos_dev[errores] == t]
    for i in rng_ej.choice(idx_t, size=min(2, len(idx_t)), replace=False):
        ... contexto, real, predicho, puesto del real, P(real), top-5
```

Elige **2 errores al azar por tipo de carácter** (5 tipos, 10 ejemplos en total, que es el mínimo de la consigna) con semilla fija, y muestra cada uno con su top-5. `replace=False` evita repetir ejemplos.

**Por qué estratificar por tipo:** si tomáramos 10 errores al azar, casi todos serían vocales y consonantes (son la mayoría), y no veríamos los errores de mayúsculas y puntuación.

---

#### Celda 11 — PCA de los embeddings

```python
desde_cero = max(["E1 ...", "E2 ..."], key=dev_acc_mean)
M = modelos[desde_cero].get_layer("embedding").get_weights()[0]
pca = PCA(n_components=2, random_state=SEED); Z = pca.fit_transform(M)
```

**PCA** (análisis de componentes principales) proyecta vectores de muchas dimensiones a 2, conservando la mayor varianza posible. El título informa cuánta varianza explican esas 2 componentes: si es baja, el gráfico es una vista parcial.

**Qué buscar:** si los caracteres del mismo tipo (vocales, consonantes, mayúsculas, puntuación) quedan agrupados, el embedding capturó su **función** en el texto. Por ejemplo, que las vocales aparezcan juntas en el aprendido desde cero significa que la red descubrió sola que las vocales "se comportan parecido".

`get_layer("embedding").get_weights()[0]` extrae la tabla aprendida. Para eso la capa se llamó `name="embedding"` en el constructor.

---

#### Celda 12 — Checklist de la Etapa 3

Imprime los requisitos de la 4.3.B cumplidos y la mejora sobre la Etapa 2.

---

#### Sección 8 (celdas 13 a 16) — Embeddings de palabras: pre-entrenados vs desde cero

**Por qué existe:** la consigna 4.3.B pide **embeddings de palabras** (GloVe/Word2Vec) y comparar pre-entrenados contra desde cero. Nuestra tarea es de caracteres y GloVe no tiene vectores de caracteres, así que la comparación se hace en una tarea **aparte**: predecir la **siguiente palabra**. No compite con el resto: es otra tarea, con otra cantidad de clases.

**Celda 13 — Tokenización y ventanas:**

```python
patron = re.compile(r"[a-z]+|'[a-z]+|[0-9]+|[^\sa-z0-9]")
tokens_por_bloque = {b: patron.findall(text[ini:fin].lower()) for b, (ini, fin) in enumerate(limites)}
```

- La regex corta en: palabras, sufijos con apóstrofo (`'s`), números, o **cualquier otro carácter suelto** que no sea espacio (la puntuación). `[^\sa-z0-9]` significa "un carácter que no sea espacio, letra ni dígito".
- El posesivo se separa (`hill's` → `hill` + `'s`) porque así está tokenizado el corpus de GloVe, y entonces más palabras encuentran vector.
- Cada bloque de la Etapa 1 (`text[ini:fin]`) se corta en palabras **por separado**: **misma partición, sin fuga**. Una palabra que cruza el borde entre dos bloques queda partida en dos, un efecto mínimo con 50 bloques (4.087 tokens en total).
- Ventanas de **N = 3 palabras** → la siguiente, armadas dentro de cada bloque.
- Vocabulario = palabras de Train + `<UNK>` (índice 0). Una palabra de Dev que no está en Train se mapea a `<UNK>` y **cuenta siempre como error**.
- Imprime cuántas palabras de Dev no aparecen en Train: eso es un **techo** de accuracy.

**Celda 14 — GloVe 6B:** lee `glove.6B.100d.txt` **directamente del zip** (sin descomprimir) y guarda solo los vectores de nuestro vocabulario. GloVe 6B se entrenó con Wikipedia 2014 + Gigaword 5, tiene 400.000 palabras en minúsculas (Pennington, Socher y Manning, 2014). Las palabras sin vector arrancan al azar, con la misma escala (desvío) que los vectores de GloVe.

**Celda 15 — Los tres modelos:** P0 (desde cero), P1 (GloVe congelado) y P2 (GloVe ajustable), con la misma red densa 128-64-32, 5 semillas y Early Stopping en los tres. Reporta accuracy **top-1** y **top-5** en Dev:
- **Top-5:** se cuenta como acierto si la palabra real está entre las 5 más probables. Con ≈ 1.000 clases, la top-1 es muy exigente.
- La función `acc_topk` hace tres pasos comentados: (1) `np.argsort(P, axis=1)[:, ::-1][:, :k]` se queda con las k columnas más probables de cada fila; (2) `(topk == y[:, None]).any(axis=1)` pregunta si la real está entre ellas (`y[:, None]` convierte el vector en columna para comparar por *broadcasting*); (3) `en_topk & (y != 0)` hace que `<UNK>` (índice 0) cuente siempre como error.
- **Ojo con P0:** en 4 de las 5 semillas da exactamente lo mismo que predecir siempre *the* (7,63 %). Early Stopping recupera los pesos de una época en la que el modelo solo aprendió la palabra más frecuente.

**Celda 16 — Vecinos más cercanos:** para algunas palabras, las 5 más parecidas por **similitud coseno** en GloVe y en el embedding aprendido desde cero.
$$\cos(u, v) = \frac{u \cdot v}{\|u\| \|v\|}$$
GloVe trae relaciones de significado aprendidas de miles de millones de palabras (`sea` → `waters, shore…`). El aprendido desde cero, con ≈ 2.800 ventanas, no llega a aprender significado (`sea` → `taverns, to, everything…`). **Este contraste es el argumento central de "pre-entrenado vs desde cero":** el pre-entrenado transfiere conocimiento que nuestros datos no alcanzan a enseñar.

---

## Parte 2b. Notebook de la Etapa 5

#### Celda 0 — Configuración

Los mismos imports y `SEEDS` de la Etapa 3, más `re` y `Counter` para contar palabras en la sección 4.

---

#### Celda 1 — Datos, ahora con Test

```python
def ventanas_de_bloques(ids):
    ...
    return X, y, np.array(b_de)       # además devuelve el bloque de cada ventana
X_test, y_test, bloque_test = ventanas_de_bloques(bloques_split["test"])
```

La misma función de la Etapa 3, que además devuelve **de qué bloque** sale cada ventana, para medir la accuracy por bloque. Es la primera vez en el trabajo que se arman las ventanas de Test: 2.520, de 7 bloques.

---

#### Celda 2 — Resultados guardados y configuración del ganador

```python
GANADOR = res3["ganador"]
cfg_g = rg.get("config_lstm", {"dropout": cfg3["dropout"], "lr": cfg3["lr"]})
```

Lee los JSON de las Etapas 1, 2 y 3. Toma la configuración del ganador de la Etapa 3 de lo que se guardó ahí (`config_lstm` para R3 y R4), así no se escribe nada a mano. Para R4: 5 caracteres, embedding de 300, Dropout 0,05, lr 0,0005, `clipnorm` 1,0, Early Stopping con paciencia 5.

---

#### Celda 3 — Reentrenamiento y evaluación única en Test

```python
for s in SEEDS:
    keras.utils.set_random_seed(s)
    modelo = construir_lstm(k, c["emb_dim"], c["dropout"])
    ... fit igual que en la Etapa 3 ...
    aciertos_dev.append(p_dev.argmax(axis=1) == yi_dev)
    aciertos_test.append(p_test.argmax(axis=1) == yi_test)
dif = abs(df_test["dev_acc"].mean() - rg["dev_acc_mean"]) * 100
```

- **Por qué reentrenar y no guardar los pesos:** con semillas fijas el entrenamiento es reproducible. La celda lo verifica: el Dev reproducido coincide con el de la Etapa 3 (diferencia 0,00 pp). Así sabemos que el modelo evaluado en Test es exactamente el elegido.
- `construir_lstm` arma las capas en el mismo orden que `construir_modelo` de la Etapa 3. Eso importa: con la misma semilla, las mismas capas en el mismo orden reciben los mismos pesos iniciales.
- Guarda, por semilla, qué ventanas acertó en Dev y en Test (`aciertos_dev`, `aciertos_test`). Con eso se calculan después todas las accuracy por subgrupo sin volver a predecir.
- **Es la única celda del trabajo que usa Test.**

---

#### Celda 4 — Test contra Dev

Tabla con accuracy (media ± desvío), peor semilla, pérdida y perplejidad en Dev y en Test, y la accuracy de cada bloque de Test. `aciertos_test[:, bloque_test == b].mean()` promedia las 5 semillas en las ventanas del bloque `b`.

---

#### Celda 5 — Tabla comparativa (4.5.1)

Arma la tabla de todos los modelos (baseline, ganador de la Etapa 2 y los de la Etapa 3) con arquitectura, parámetros, Dev, Test (solo el ganador) y tiempo por corrida, leyendo todo de los JSON. Grafica tiempo (escala logarítmica) contra accuracy en Dev: muestra que, después del MLP, cada punto de accuracy cuesta mucho más tiempo y parámetros.

---

#### Celda 6 — Sesgos medibles en el texto (4.5.3)

```python
palabras = re.findall(r"[a-z]+", text.lower())
GENERO = {"masculino": ["he", "him", "his", ...], "femenino": ["she", "her", "hers", ...]}
hapax = sum(1 for w, n in frec.items() if n == 1)
```

Cuenta pronombres y sustantivos de persona por género (52 masculinos contra 10 femeninos) y las palabras que aparecen una sola vez (*hapax*: el 69,5 % del vocabulario). Son sesgos que se pueden **medir** en el texto, en lugar de suponerlos.

---

#### Celda 7 — Desempeño por subgrupos en Test (4.5.3)

```python
ranking = [c for c, _ in Counter(y_train).most_common()]
def grupo_frecuencia(c): ...   # top 10 / puestos 11-30 / puesto 31 o más
```

Accuracy en Dev y en Test por tipo de carácter y por frecuencia del carácter en Train. La agrupación por frecuencia viene del trabajo de Thiago (Parte 5). El patrón: cuanto menos aparece un carácter en Train, peor lo predice el modelo (53 %, 27 % y 2 % en Test).

---

#### Celda 8 — Persistencia

Guarda el ganador, su configuración, cada corrida y los subgrupos en `resultados_etapa5.json`.

---

## Parte 3. Conceptos transversales

### Sesgo, varianza y lo que hay que hacer (Ng, *Machine Learning Yearning*)

| Diagnóstico | Síntoma | Qué hacer |
|---|---|---|
| **Sesgo alto** (underfitting) | Error alto en Train **y** en Dev, brecha chica | Modelo más grande o expresivo, mejores features, menos regularización |
| **Varianza alta** (overfitting) | Error bajo en Train, alto en Dev, brecha grande | Más datos, regularización, modelo más chico, Early Stopping |
| **Data mismatch** | Dev y Test de otra distribución que Train | Conseguir datos de Train parecidos a Dev; medir con un "train-dev set" |
| **Error irreducible** (Bayes) | Error que ningún modelo puede bajar con esa información | Darle al modelo más información (más contexto) |

**Nuestro recorrido:** baseline → **sesgo** → MLP → **varianza** → regularización → mejora → embeddings y LSTM → la varianza sigue y el techo se mantiene → **el límite son los datos**.

### Accuracy, pérdida y perplejidad

- **Accuracy top-1:** fracción de aciertos del carácter más probable. Es nuestra **métrica de número único**, porque es fácil de interpretar y de comparar con la referencia trivial.
- **Entropía cruzada:** $-\frac{1}{N}\sum \log p(\text{real})$. Es lo que **optimiza** la red, y castiga la confianza mal puesta.
- **Perplejidad:** $e^{\text{loss}}$, el "número efectivo de opciones" entre las que duda el modelo.
- Pueden **no coincidir**: un modelo puede mejorar la accuracy y empeorar la perplejidad (sobreconfianza), o al revés. En la Etapa 3, R1 y R4 empatan en accuracy (49,69 % y 50,07 %), pero R1, que sigue entrenando hasta la época 100, tiene perplejidad 7,64 y R4, cortado a tiempo, 5,91.

### Margen de error en Dev

Con $n = 2.880$ y accuracy $p \approx 0{,}41$, el error estándar es $\sqrt{p(1-p)/n} \approx 0{,}0092$, así que el intervalo al 95 % es ≈ $\pm 1{,}8$ pp. **Cualquier diferencia menor a eso entre dos modelos es difícil de defender como mejora real.** El desvío entre semillas mide otra cosa: cuánto cambia el resultado por la inicialización, con el Dev fijo.

### Por qué Test no se toca

Cada vez que elegimos algo (hiperparámetros, ganador, época de corte) mirando Dev, el número de Dev queda levemente optimista: elegimos lo que "le fue bien" a ese conjunto en particular. Test se evalúa **una sola vez**, al final (Etapa 5), como medición no sesgada. En nuestro caso, el ganador baja de 50,07 % en Dev a 45,70 % en Test.

### Embeddings: congelado contra ajustable

- **Congelado:** los vectores no cambian, y la red solo aprende a usarlos. Funciona si los vectores ya codifican lo que la tarea necesita.
- **Ajustable (fine-tuning):** los vectores son el punto de partida y se actualizan con el gradiente. Con datos suficientes, la red los reescribe para la tarea.
- **Desde cero:** se aprenden solo con nuestros datos.

---

## Parte 4. Preguntas probables de la defensa

**1. ¿Por qué partieron por bloques y no al azar?**
Porque las ventanas se solapan en 39 de 40 caracteres. Al azar, Dev tendría casi copias de ejemplos de Train (fuga de datos) y la accuracy saldría inflada. Con bloques de 400 caracteres asignados al azar, Dev y Test son pedazos de texto que el modelo nunca vio, y los tres conjuntos cubren todo el cuento.

**2. ¿Por qué accuracy y no F1-macro?**
Porque es interpretable y comparable con la referencia trivial. F1-macro le daría el mismo peso a un carácter que aparece una vez que al espacio, y la métrica quedaría dominada por clases con 1 o 2 casos en Dev. Igual analizamos los errores por tipo de carácter para no esconder el desbalance.

**3. ¿Cómo saben que el baseline tiene sesgo y no varianza?**
Error de Train ≈ 57 % y de Dev ≈ 59 %: error alto en los dos y brecha chica (≈ 2 pp). Si fuera varianza, Train estaría mucho mejor que Dev.

**4. ¿Por qué el MLP usa los mismos 5 caracteres que el baseline?**
Para aislar el efecto de la no linealidad. Si le hubiéramos dado más contexto, no sabríamos si mejoró por la red o por la información extra.

**5. ¿Cuántos parámetros tiene el MLP y cómo se calculan?**
53.536. Cada capa densa tiene $n_{in} \cdot n_{out} + n_{out}$: $(320 \cdot 128 + 128) + (128 \cdot 64 + 64) + (64 \cdot 32 + 32) + (32 \cdot 64 + 64)$.

**6. ¿Por qué Early Stopping cortaba antes de tiempo y cómo lo resolvieron?**
Mira la pérdida, y la pérdida empieza a subir (sobreconfianza) antes de que la accuracy deje de mejorar. En el MLP, Early Stopping solo quedó en 45,75 % y no entró en el ganador. En la Etapa 3 sí funcionó con la LSTM de 5 caracteres (R4): con lr 0,0005, cortar en el mínimo de la pérdida da la misma accuracy que entrenar 100 épocas, con mucho menos sobreajuste. Probamos también ReduceLROnPlateau para alinear las curvas; lo sacamos porque no se vio en clase (`docs/EXPERIMENTOS_RETIRADOS.md`).

**7. ¿Por qué combinaron las técnicas de a una?**
Para saber cuánto aporta cada una y detectar las que se pisan (Early Stopping con L2, Dropout con BatchNorm). Una técnica se acepta solo si mejora más que el desvío entre semillas.

**8. ¿El 50,07 % de la Etapa 3 es mejor que el 48,71 % de la Etapa 2?**
La diferencia (+1,36 pp) es menor que el margen de error de Dev (± 1,8 pp) y apenas mayor que el desvío entre semillas. Lo honesto es decir que es una mejora chica, no concluyente. Lo importante es la explicación: modelos muy distintos chocan con el mismo techo y todos con varianza alta, así que el cuello de botella son los datos (≈ 12.800 ventanas de entrenamiento).

**9. ¿Por qué la LSTM de 40 caracteres anda peor que la de 5?**
Sin Early Stopping sobreajusta: ver más contexto le da más formas de memorizar Train (R2, 45,06 %). Con Early Stopping se recupera a 47,76 % (R3), pero con el mismo esquema la de 5 la supera por 2,31 pp (R4). Con el lr de 0,005 pasa lo mismo. Con tan pocos datos, el contexto largo no llega a aprovecharse.

**10. ¿Si R1 y R4 empatan, por qué el ganador es R4?**
El criterio fijado es la mayor accuracy media en Dev, y R4 la tiene (50,07 % contra 49,69 %). La diferencia es chica, pero R4 además tiene menos brecha (10,43 contra 28,67 pp), mejor perplejidad (5,91 contra 7,64) y tarda la mitad. Es el mismo modelo que R1, cortado en el mínimo de la pérdida.

**11. La consigna pedía embeddings de palabras. ¿Por qué usaron de caracteres?**
Porque la tarea es predecir caracteres, y cada token de entrada es un carácter. Para la comparación pre-entrenados vs desde cero que pide la consigna usamos palabras (sección 8, siguiente palabra con GloVe 6B), porque GloVe no tiene vectores de caracteres. Ahí GloVe ayuda en promedio (10,81–11,72 % de top-1 contra 8,08 % desde cero), aunque con mucho desvío entre semillas. También probamos vectores de caracteres derivados de GloVe de un repositorio externo; no aportaban y los sacamos (`docs/EXPERIMENTOS_RETIRADOS.md`).

**12. ¿Qué harían con más tiempo o recursos?**
Más datos (otros cuentos de Lovecraft para entrenar, dejando Dev y Test solo con *The Festival*); validación cruzada por bloques, para no depender de qué bloques tocaron en Dev; y GPU para probar la LSTM de 40 caracteres con más configuraciones.

**13. ¿Qué es la perplejidad y por qué la reportan?**
$e^{\text{loss}}$, el número efectivo de opciones entre las que duda el modelo. Complementa la accuracy: mide la calidad de las probabilidades, no solo el top-1.

**14. ¿Usaron Test en algún momento?**
Solo una vez, en la Etapa 5, con el ganador elegido por Dev. En las Etapas 1 a 3 ni siquiera se generan sus ventanas.

**15. ¿Por qué el resultado en Test es 4,37 pp peor que en Dev?**
Es unas 8 veces el desvío entre semillas, así que no es ruido del entrenamiento. Hay dos explicaciones que no podemos separar: (1) elegimos entre muchos modelos mirando Dev, así que el número de Dev del ganador es optimista; (2) las partes del cuento son distintas entre sí: en Test, la accuracy por bloque va de 43,00 % a 50,44 %, y con 7 u 8 bloques por conjunto qué bloques tocaron mueve la media.

**16. ¿Por qué evaluaron solo el ganador en Test?**
La consigna 4.5.2 pide evaluar el mejor modelo en Test por primera y única vez. Evaluar todos sería usar Test para comparar, que es el trabajo de Dev. La contra: no sabemos si el orden entre modelos se mantiene en Test.

---

## Parte 5. Enfoque alternativo de Thiago (rama `dev_thiago`)

> Esta parte la explica Thiago. Es una segunda forma de encarar las Etapas 1, 2, 3 y 5 que probamos en paralelo, con una estructura distinta a la de `main` (que hoy es la de las Partes 1 y 2). Las cifras salen de las salidas guardadas en `notebooks/stage_3_lstm.ipynb` y `notebooks/stage_5_cierre.ipynb` de esa rama.

### Qué cambia respecto de `main`

| Aspecto | `main` | `dev_thiago` |
|---|---|---|
| Organización del código | todo en el notebook | módulos en `src/` (datos, modelos, entrenamiento, gráficos); el notebook orquesta |
| Partición | 50 bloques 35/8/7, ventanas con stride 1 | corte contiguo 80/10/10 (Test = último 10 % del cuento) |
| Ventanas | 40 caracteres, se predice el siguiente | secuencias de 48 caracteres con stride 5 en Train y disjuntas en Dev y Test; la LSTM predice cada posición (*seq2seq*) |
| Métrica principal | accuracy en Dev, 5 semillas | perplejidad en Dev, 1 semilla |
| LSTM | Embedding(300) → LSTM(128) → softmax | Embedding(50) → LSTM(96) → Dense(48) → logits, 67.638 parámetros |
| Embeddings pre-entrenados | GloVe 6B 100d con palabras (sección 8) | GloVe 50d vía `gensim`, como inicialización del embedding de caracteres (`lstm_combined_glove`) |
| Regularización en la Etapa 3 | Early Stopping (esquema de la clase 8) | Dropout, L2, L1/L2 (ElasticNet), Early Stopping y ReduceLROnPlateau |

### Resultados de esa rama (su partición 80/10/10)

| Etapa | Modelo | Parámetros | Dev PPL | Dev Acc |
|---|---|---|---|---|
| 1 | Baseline (regresión logística, K=5) | 21.515 | 8,74 | 38,11 % |
| 2 | MLP base | 54.882 | 7,47 | 44,22 % |
| 2 | MLP + Dropout 0,05 | 54.882 | 7,04 | 46,80 % |
| 3 | `lstm_baseline` | 67.638 | 6,99 | 45,12 % |
| 3 | `lstm_reg_dropout` | 67.638 | 6,43 | 45,99 % |
| 3 | `lstm_combined_glove` | 67.638 | 6,48 | 45,93 % |
| 3 | `lstm_combined_scratch` | 67.638 | 6,42 | 46,49 % |
| 3 | `lstm_combined_l1_l2` (ganador) | 67.638 | 6,33 | 46,75 % |

El ganador se evaluó en su Test: perplejidad 7,46 y accuracy 44,97 % (−1,78 pp respecto de su Dev).

### Qué aporta y dónde quedó en `main`
- **Estructura de la Etapa 5** según la consigna 4.5.1 a 4.5.4: la tomamos tal cual en `notebooks/ETAPA_5.ipynb`.
- **Desempeño por frecuencia del carácter** (los 10 más frecuentes, puestos 11 a 30, el resto) como subgrupo del análisis ético: está en la sección 4.2 de la Etapa 5.
- **La perplejidad con L1/L2:** en Keras, `val_loss` suma la penalización de los pesos, así que $e^{\text{val\_loss}}$ exagera la perplejidad de los modelos regularizados. Thiago la recalculó solo con la entropía cruzada. En `main` no usamos L1/L2 en la Etapa 3, pero es un buen argumento si preguntan por qué la pérdida de la Etapa 2 con L2 no se compara directo.
- **Ventana deslizante solo en Train:** fue lo que llevó a discutir el stride en el grupo; en `main` quedó stride 1 dentro de los bloques.

### Por qué los números no se comparan con los de `main`
- **Otra partición:** su Dev es un tramo contiguo de 2.020 caracteres (41 secuencias disjuntas); el nuestro son 8 bloques repartidos por el cuento, con 2.880 ventanas.
- **Otra tarea de salida:** su LSTM predice las 48 posiciones de cada secuencia, incluidas las primeras, que tienen muy poco contexto; la nuestra predice solo el carácter 41.
- **Una semilla contra cinco:** con un Dev tan chico, la diferencia entre sus modelos de la Etapa 3 (45,12 % a 46,75 %) queda dentro de lo que puede variar entre semillas.

---

## Parte 6. Más datos: la versión multi-cuento de Brian (rama `dev_brian`)

> Esta parte la explica Brian. Responde a lo que señalaban los diagnósticos de las Etapas 2 y 3: faltan datos. Las cifras salen de las salidas guardadas en `notebook/ETAPA_1_MULTICUENTO.ipynb` de esa rama.

### Qué propone

| Aspecto | `main` | `dev_brian` |
|---|---|---|
| Corpus | *The Festival* (20.180 caracteres) | varios cuentos de Lovecraft; en la versión subida, *The Call of Cthulhu* (69.904) y *The Festival* (90.084 en total) |
| Test | 7 bloques de *The Festival* | cuentos **enteros** reservados |
| Train y Dev | bloques de *The Festival* | bloques de los cuentos no reservados |
| Vocabulario | 64 caracteres | 80 caracteres |

**Qué suma al análisis:**
- **Data mismatch entre cuentos**, medido con la divergencia de Jensen-Shannon entre las distribuciones de caracteres: 0,0055 entre los dos cuentos, o sea, casi el mismo vocabulario de caracteres.
- **Verificación anti-fuga**: busca fragmentos de 50 caracteres repetidos entre los cuentos de Test y los de Train/Dev (0 coincidencias).
- **Grilla de K más amplia** en el baseline ({1, 3, 5, 7, 10}): con más datos, el mejor K pasa de 5 a 7.

### Resultado de su Etapa 1

Con dos cuentos, *The Festival* quedó entero en Test y Train/Dev salieron solo de *The Call of Cthulhu*: 56.424 ventanas de Train y 6.480 de Dev. El baseline (K = 7) llega a 38,44 % en Dev, con brecha de 2,43 pp. No se compara con nuestro 40,83 %, porque su Dev es otro cuento.

### Cómo lo usamos en `main`

Su enfoque completo implica rehacer todas las etapas con otra partición. Tomamos la idea central, **más datos para entrenar**, en un experimento acotado: *The Call of Cthulhu* se suma **solo a Train**, Dev sigue siendo el de *The Festival* y R4 se reentrena con 3 semillas. Se mide solo en Dev, porque Test ya se usó en la Etapa 5. Texto: Project Gutenberg #68283, en `db/The_Call_of_Cthulhu.txt`; solo entran las ventanas con caracteres del vocabulario de *The Festival* (67.212 de 69.810).

| Train | Ventanas | Dev (3 semillas) | Perplejidad |
|---|---|---|---|
| Solo *The Festival* | 12.780 | 50,31 % ± 0,48 % | 5,89 |
| *The Festival* + *Cthulhu* | 79.992 | **54,40 % ± 0,58 %** | 4,92 |

**+4,09 pp en Dev**, más que cualquier cambio de arquitectura del trabajo: confirma que el límite eran los datos. Está en `experimentos/04_mas_datos_Call_of_Cthulhu.ipynb` y en la sección 2 de la Etapa 5 ("Mejoras posibles").

---

## Sección R. Resultados para tener en la cabeza

> Cifras de las corridas locales del 21/11/2026 (CPU, 5 semillas, Dev salvo donde dice Test). Son las que están en los notebooks.

### Datos y partición
- *The Festival*: 20.180 caracteres después de la limpieza, vocabulario de 64 caracteres, 3.578 palabras.
- 50 bloques de unos 400 caracteres: 35 / 8 / 7 → 12.780 / 2.880 / 2.520 ventanas (Train / Dev / Test).
- Clase mayoritaria: el espacio, con el 17,85 % de Dev.

### La línea de resultados (accuracy en Dev)

| Paso | Modelo | Dev | Mejora | Diagnóstico |
|---|---|---|---|---|
| Referencia | Predecir siempre espacio | 17,85 % | — | — |
| Etapa 1 | Regresión logística, one-hot de 5 caracteres | 40,83 % | +22,99 pp sobre la referencia | sesgo alto (Train 42,97 %, brecha 2,13 pp) |
| Etapa 2 | MLP 128-64-32 sin regularizar | 46,90 % ± 0,49 % | +6,07 pp | varianza alta (Train 71,07 %, brecha 24,17 pp) |
| Etapa 2 | **MLP + Dropout 0,05** | **48,71 % ± 0,52 %** | **+7,88 pp sobre el baseline** | sigue la varianza (brecha 20,72 pp) |
| Etapa 3 | **R4 — LSTM de 5 caracteres + Early Stopping** | **50,07 % ± 0,54 %** | **+1,36 pp sobre la Etapa 2** | brecha de 10,43 pp |
| Etapa 5 | R4 en **Test** (única evaluación) | **45,70 % ± 0,38 %** | −4,37 pp respecto de Dev | — |

### Números de la Etapa 2 que conviene recordar
- Learning rate elegido: 0,0005 (pico de 47,56 %, contra 47,26 % de 0,001).
- Dropout 0,05: +1,81 pp sobre el MLP Base, y es la única técnica que supera el umbral (0,49 pp).
- L2: sin efecto (de 46,66 % a 47,08 %). Early Stopping: 45,75 % (corta antes del pico). Batch Normalization: 43,10 %, con perplejidad de 252.
- Ninguna combinación con Dropout mejora: con L2 48,70 %, con Early Stopping 47,00 % y con BatchNorm 48,32 %.
- Error de Train: baseline 57,03 % → ganador 30,57 % (sesgo evitable de al menos 26,46 pp).

### Números de la Etapa 3 que conviene recordar

| Modelo | Dev | Brecha | Perplejidad | Parámetros | Tiempo por corrida |
|---|---|---|---|---|---|
| E0 — One-hot (= ganador de la Etapa 2) | 48,71 % ± 0,52 % | 20,72 pp | 7,66 | 53.536 | 35 s |
| E1 — Desde cero, 16 dim. | 47,88 % ± 0,73 % | 10,11 pp | 6,51 | 23.840 | 25 s |
| E2 — Desde cero, 300 dim. | 49,51 % ± 0,81 % | 25,62 pp | 8,66 | 223.776 | 40 s |
| R1 — LSTM, 5 caracteres | 49,69 % ± 0,12 % | 28,67 pp | 7,64 | 247.104 | 100 s |
| R2 — LSTM, 40 caracteres | 45,06 % ± 1,70 % | 27,41 pp | 8,54 | 247.104 | 583 s |
| R3 — LSTM 40 + ES | 47,76 % ± 0,55 % | 9,20 pp | 6,16 | 247.104 | 284 s |
| **R4 — LSTM 5 + ES (ganador)** | **50,07 % ± 0,54 %** | 10,43 pp | **5,91** | 247.104 | 45 s |

- R3 y R4: Adam con `clipnorm` 1,0, lr 0,0005 y Early Stopping con paciencia 5 (47 a 53 épocas).
- Primer carácter de una palabra: menos del 20 % de acierto. Desde la segunda letra, más del 48 %.
- Mayúsculas: 16 % (38 casos). Puntuación: 3 % (60 casos).
- El ganador falla en el 50,00 % de Dev.

### Experimento de palabras (sección 8)

| Modelo | Top-1 | Top-5 | Parámetros |
|---|---|---|---|
| Siempre *the* | 7,63 % | — | — |
| P0 — Desde cero | 8,08 % ± 1,02 % | 23,54 % | 181.864 |
| P1 — GloVe congelado | 10,81 % ± 2,84 % | 25,49 % | 81.864 |
| P2 — GloVe ajustable | 11,72 % ± 2,76 % | 25,58 % | 181.864 |

- El 25,2 % de las palabras de Dev no aparece en Train, así que el techo de accuracy es 74,8 %.
- GloVe tiene vector para 959 de las 999 palabras del vocabulario.
- P0 da exactamente 7,63 % (lo mismo que "siempre *the*") en 4 de las 5 semillas.
- Limitación: Early Stopping corta en unas 11 épocas, cuando la accuracy en Dev todavía sube, así que las cifras subestiman a los tres modelos.

### Números de la Etapa 5 que conviene recordar
- R4 en Test: **45,70 % ± 0,38 %** (peor semilla 45,12 %), perplejidad 7,10. En Dev: 50,07 %, perplejidad 5,91.
- Accuracy por bloque de Test: de 43,00 % a 50,44 %.
- Test por tipo: espacio 86 %, vocales 40 %, consonantes 38 %, mayúsculas 10 %, puntuación 3 %.
- Test por frecuencia del carácter: top 10 53 %, puestos 11 a 30 27 %, puesto 31 o más 2 %.
- Texto: 52 palabras masculinas contra 10 femeninas; el 69,5 % del vocabulario aparece una sola vez.

### Las tres ideas para defender
1. **Cada etapa responde a un diagnóstico:** sesgo → MLP; varianza → regularización; techo con 5 caracteres → embeddings y LSTM.
2. **Las mejoras de la Etapa 3 son chicas** (+1,36 pp, dentro del margen de ±1,8 pp de Dev). Modelos muy distintos chocan con el mismo techo: el cuello de botella son los datos. Test lo confirma: 45,70 %.
3. **Cuándo parar importa tanto como la arquitectura:** R4 es R1 cortado a tiempo, con la misma accuracy y mucho menos sobreajuste.

---

## Glosario rápido

| Término | Significado |
|---|---|
| Ventana / contexto | Los $L = 40$ caracteres que el modelo recibe como entrada |
| Stride | Cuánto se corre la ventana entre una muestra y la siguiente (acá, 1) |
| One-hot | Vector de ceros con un 1 en la posición de la clase |
| Embedding | Vector denso aprendido (o pre-entrenado) que representa un token |
| Softmax | Convierte números reales en probabilidades que suman 1 |
| Época | Una pasada completa por Train |
| Mini-batch | Grupo de muestras (128) con el que se hace cada actualización de pesos |
| Learning rate | Tamaño del paso del optimizador |
| Brecha Train–Dev | Accuracy de Train menos accuracy de Dev; mide la varianza |
| pp | Puntos porcentuales (de 40 % a 48 % son +8 pp) |
| Ablación | Sacar o agregar componentes de a uno para medir su aporte |
| Top-k | Acierto si la clase real está entre las k más probables |
| OOV / `<UNK>` | Palabra fuera del vocabulario de entrenamiento |
| PCA | Proyección lineal a pocas dimensiones que conserva la mayor varianza |
| Fine-tuning | Seguir entrenando vectores (o pesos) pre-entrenados con la tarea propia |
