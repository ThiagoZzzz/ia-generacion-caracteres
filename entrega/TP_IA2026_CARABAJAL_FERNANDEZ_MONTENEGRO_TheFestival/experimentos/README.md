# Experimentos e historial

Lo que probamos y no quedó en el notebook principal. Cada notebook conserva las salidas de su corrida; del texto quedan los títulos y un encabezado con el resultado. Accuracy en Dev, media ± desvío entre semillas.

| Notebook | Qué se probó | Resultado | Por qué no quedó |
|---|---|---|---|
| `00a_Etapa_1_primera_version` | Naive Bayes de pasajes; ventanas al azar con paso 5; baseline sobre n-gramas | baseline 19,53 % | fuga entre conjuntos y vocabulario recortado |
| `00b_Etapas_1_y_2_version_anterior` | regularización elegida con una grilla combinada | MLP + Dropout 0,1: 48,44 % ± 0,34 % | se rehízo con ajuste del lr y técnicas de a una |
| `01_Etapas_1_y_2_con_ReduceLROnPlateau` | ReduceLROnPlateau (suave y agresivo); lr 0,0001 con 300 épocas | alinea las curvas, no sube el techo; con Dropout, 47,80 % | rinde menos que el lr fijo (48,71 %) |
| `02_Etapa_3_con_Plateau_y_embeddings_de_caracteres` | embeddings de caracteres derivados de GloVe 840B; Plateau en R3 y R4; variantes de R3 | congelado 38,29 %, ajustable 49,06 %; R3 + Dropout 0,2: 49,06 %; R3 + emb. 16: 45,94 % | GloVe no tiene vectores de caracteres; ninguna variante mejora más que el desvío |
| `03_R4_learning_rate_y_paciencia` | R4 con lr 0,005 o 0,0005 y paciencia 5 o 10 | 49,20 % con 0,005; 50,07 % con 0,0005; la paciencia no cambia nada | quedó lr 0,0005 y paciencia 5 |
| `04_mas_datos_Call_of_Cthulhu` | *The Call of Cthulhu* sumado a Train (3 semillas) | 50,31 % → **54,40 %** | Test ya estaba usado; queda como evidencia de que el límite son los datos |
| `05_Etapa_3_particion_contigua_Thiago` | versión paralela: partición contigua 80/10/10, LSTM que predice cada posición, L1/L2, 1 semilla | Dev 46,75 %, perplejidad 6,33 | otra partición y otra tarea de salida: no se compara |
| `06_Etapa_5_particion_contigua_Thiago` | su cierre: Test, tabla comparativa, ética por subgrupos | Test 44,97 % (en su partición) | se tomaron su estructura, su análisis por subgrupos y sus ideas |

**Cambios de código sin efecto en las cifras:** el one-hot del MLP pasó de una matriz armada a mano a `to_categorical` (matriz idéntica), y la tokenización de palabras de la Etapa 3 se hace por bloque (de 4.057 a 4.087 tokens).

Los notebooks 03 y 04 corren desde esta carpeta con los datos de la entrega (`../db/`, `../data/processed/`). Los 00a a 02 son versiones anteriores completas, y los 05 y 06 necesitan los módulos `src/` de la rama `dev_thiago`: quedan como registro y pueden no correr con los archivos actuales.
