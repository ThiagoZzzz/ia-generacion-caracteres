# TP Final IA 2026 — Predicción del siguiente carácter en *The Festival* (H. P. Lovecraft)

**Grupo 9:** Leonardo Carabajal, Brian Fernandez, Thiago Montenegro Yerumini · **Docente:** Pablo Moreira · UNAB

Modelo que predice el carácter siguiente a partir de los 40 anteriores. El notebook recorre las Etapas 1, 2, 3 y 5 de la consigna; la Etapa 4 (optativa) no se realizó.

| Etapa | Mejor modelo | Accuracy en Dev |
|---|---|---|
| 1 | Regresión logística sobre los últimos 5 caracteres | 40,83 % |
| 2 | MLP 128-64-32 con Dropout 0,05 | 48,71 % ± 0,52 % |
| 3 | LSTM de 5 caracteres con Early Stopping (R4) | 50,07 % ± 0,54 % |
| 5 | R4 evaluado **una única vez** en Test | **45,70 % ± 0,38 %** |

## Contenido

```
TP_IA2026_CARABAJAL_FERNANDEZ_MONTENEGRO_TheFestival/
├── TP_IA2026_CARABAJAL_FERNANDEZ_MONTENEGRO_GRUPO9.ipynb   notebook principal (con salidas)
├── requirements.txt                                          librerías y versiones
├── README.md
├── db/
│   ├── The_Festival.txt                                      texto original del dataset
│   └── The_Call_of_Cthulhu.txt                               Project Gutenberg #68283 (solo para un experimento)
├── data/processed/                                           particiones, resultados de cada etapa y caché de R2
├── resultados/                                               gráficos exportados
└── experimentos/                                             lo que probamos y no quedó en el notebook principal
```

## Entorno

Probado con Python 3.12.10 en Windows, solo CPU.

```bash
python -m venv venv
venv\Scripts\activate          # en Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

## Ejecución

Abrir el notebook desde esta carpeta (la raíz del proyecto) y ejecutar todas las celdas en orden. Las rutas son relativas a esta carpeta: el texto se lee de `db/The_Festival.txt` y los resultados se guardan en `data/processed/`.

Tiempos de referencia en CPU:

| Etapa | Tiempo |
|---|---|
| 1 y 2 | unos 47 min |
| 3 | unos 45 min con el caché de R2; unos 95 min sin él |
| 5 | unos 4 min |

- **GloVe:** la sección 8 de la Etapa 3 descarga `glove.6B.zip` (unos 860 MB) a `data/external/` la primera vez.
- **Caché de R2:** la LSTM de 40 caracteres (R2) tarda unos 50 minutos. Sus resultados y curvas están en `data/processed/cache_R2.json`; si se borra ese archivo, o si cambia su configuración, se vuelve a entrenar. Con las semillas fijas da los mismos números.
- **Test:** se evalúa una sola vez, en la Etapa 5. Volver a ejecutar esa etapa repite la misma evaluación con los mismos números.
