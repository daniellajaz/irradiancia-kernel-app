# Atlas de irradiancia de Nariño

Clasificación de zonas de irradiancia solar a partir de imágenes Landsat y MODIS, comparando funciones kernel en
**1 296 pipelines** (escalador × discretizador × reductor × modelo × kernel). Los mejores se despliegan en una
aplicación web con mapas.

* **Notebook (puntos 1 a 4):** `notebooks/Clasificacion_Irradiancia_Kernel.ipynb`
* **Aplicación web (punto 5):** `app/`

## Qué hace la aplicación

| Requisito | Dónde |
|---|---|
| a) Almacenar los modelos generados | Página **Modelos guardados**. Los `.joblib` exportados por el notebook se registran solos en `models/registry.db` (SQLite) al arrancar. Desde el formulario se puede entrenar cualquier combinación de pipeline: se evalúa con el mismo protocolo del notebook y queda guardada. También se pueden eliminar modelos. |
| e) Comparar dos modelos en el mapa y en sus métricas | Página **Comparar modelos**. Tiene dos mapas sincronizados (A y B) y una regla de irradiancia con los cortes de zona de cada modelo. Muestra una tabla y un gráfico de Accuracy, F1 macro, AUC y MCC (validación cruzada ± desviación y holdout), las dos matrices de confusión, un mapa de discrepancias y el porcentaje de área en que coinciden. |
| d) Clic en el mapa → clasificación del punto | Al hacer clic en cualquiera de los mapas se muestra la zona que asigna cada modelo, con su rango de irradiancia y la distancia a la observación más cercana (con su zona real). Las bandas del punto se estiman por interpolación IDW. |

## Estructura

```
├── notebooks/Clasificacion_Irradiancia_Kernel.ipynb   # puntos 1-4 (ejecutado, con explicaciones)
├── fkernel_lab/            # librería común (la genera el notebook con %%writefile)
│   ├── original/           # KSVM.py, KANN.py, KernelUtilities.py del repositorio magohector/fkernel
│   ├── kernels.py          # kernels de Belanche vectorizados + calibración por la mediana
│   ├── estimators.py       # KSVCVec, KANNCVec (heredan de KSVC y KANNC)
│   ├── kridge.py           # KRidgeClassifier (RidgeClassifier + truco kernel)
│   ├── preprocessing.py    # discretizadores de irradiancia y reductor PCA/LDA
│   ├── experiment.py       # espacio de pipelines, evaluación, sintonización, exportación
│   └── geo.py              # EPSG:3857 ↔ WGS84, malla del mapa, interpolación IDW
├── app/                    # Flask + Leaflet + Chart.js (librerías incluidas en static/vendor)
├── models/                 # modelos .joblib exportados (+ registry.db, se crea solo)
├── data/                   # landsat_model.csv, modis_model.csv
├── results/                # resultados de la rejilla y de la sintonización
├── scripts/                # run_grid.py y tune.py (etapas costosas fuera del notebook)
├── requirements.txt  Dockerfile  Procfile
```

## Despliegue

> Los modelos se guardaron con **scikit-learn 1.8.0**. Instale las versiones de `requirements.txt` para poder cargarlos.

### 1. Local (Python 3.11 o 3.12)

```bash
git clone https://github.com/<usuario>/<repositorio>.git
cd <repositorio>
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app/app.py                  # http://127.0.0.1:5000
```

En producción use gunicorn (Linux/macOS), siempre desde la raíz del repositorio:



## Reproducir el experimento

```bash
pip install -r requirements.txt jupyter
python scripts/run_grid.py     # opcional: 1 296 pipelines (~30-45 min en 1 CPU, reanudable)
python scripts/tune.py         # opcional: sintonización de la etapa 2
jupyter notebook notebooks/Clasificacion_Irradiancia_Kernel.ipynb
```

El notebook reutiliza `results/grid_results.csv` y `results/tuning_results.json` si existen. Si no, los calcula
(o use `RECALCULAR = True`). Al final exporta los mejores pipelines a `models/`, y la app los muestra al reiniciarla.

## Notas técnicas

* Las columnas `latitude` y `longitude` de los CSV contienen coordenadas **EPSG:3857 en metros**: `latitude` es
  la X y `longitude` la Y. Se usan tal cual como variables del modelo y se convierten a grados solo para el mapa.
* Para clasificar un punto sin observación hacen falta sus 7 bandas. Se estiman por IDW con las 8 observaciones
  más cercanas del mismo satélite. Fuera del área con datos (más de 10 km) la app avisa de que es una
  extrapolación.
* Las unidades de irradiancia son las del dataset original.
* Créditos: clases KSVC/KANN y funciones kernel © Héctor Andrés Mora Paz (licencia GPL),
  [magohector/fkernel](https://github.com/magohector/fkernel). Artículo de referencia: Pachajoa, Mora‑Paz y
  Mayorca‑Torres (2021), Rev. Fac. Ing. 30(58), e13845.
