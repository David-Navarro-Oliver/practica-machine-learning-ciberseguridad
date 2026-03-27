# Práctica de Machine Learning - Clasificación de Ataques Cibernéticos

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification-0A66C2)
![Cybersecurity](https://img.shields.io/badge/Domain-Cybersecurity-37474F)
![Portfolio](https://img.shields.io/badge/GitHub-Portfolio-181717?logo=github&logoColor=white)

[🇬🇧 English version](README.en.md)

*Proyecto de clasificación multiclase en Python para predecir la acción tomada ante ataques cibernéticos a partir del dataset `cybersecurity_attacks.csv`, con enfoque académico y presentación orientada a portfolio.*

---

## 1️⃣ Resumen del proyecto

![Classification](https://img.shields.io/badge/Task-Classification-1565C0)
![Multiclass](https://img.shields.io/badge/Problem-Multiclass-6A1B9A)
![Action%20Taken](https://img.shields.io/badge/Main%20Target-Action%20Taken-C62828)

Esta práctica desarrolla un problema de clasificación multiclase sobre el dataset `cybersecurity_attacks.csv` con el objetivo de predecir la variable objetivo principal `Action Taken`. El trabajo sigue un flujo completo de análisis exploratorio, selección de variables, preprocesamiento, entrenamiento de varios modelos, evaluación y conclusión interpretada. Además, se incorporan los targets `Severity Level` y `Attack Type` como ampliación comparativa opcional, sin desplazar el foco principal del proyecto.

### Resultados clave

![Best Model](https://img.shields.io/badge/Best%20Model-Gradient%20Boosting%20Ajustado-1565C0)
![Accuracy](https://img.shields.io/badge/Accuracy-0.3410-2E7D32)
![F1 Macro](https://img.shields.io/badge/F1%20Macro-0.3372-6A1B9A)
![Dataset](https://img.shields.io/badge/Dataset-40.000%20registros%20%7C%2025%20variables-546E7A)

- **Target principal:** `Action Taken`
- **Mejor modelo:** `Gradient Boosting Ajustado`
- **Accuracy:** `0.3410`
- **F1 macro:** `0.3372`
- **Registros del dataset:** `40.000`
- **Variables:** `25`

---

## 2️⃣ Objetivos

![EDA](https://img.shields.io/badge/EDA-Included-00897B)
![Preprocessing](https://img.shields.io/badge/Preprocessing-Included-F9A825)
![Evaluation](https://img.shields.io/badge/Model%20Evaluation-Included-5E35B1)

- Analizar la estructura y calidad del dataset.
- Preparar los datos para el modelado tabular.
- Entrenar varios modelos de clasificación.
- Comparar el rendimiento entre enfoques distintos.
- Interpretar los resultados con criterio metodológico.
- Ampliar el análisis con targets adicionales comparables.

---

## 3️⃣ Dataset

![Dataset](https://img.shields.io/badge/Dataset-cybersecurity__attacks.csv-546E7A)
![Registros](https://img.shields.io/badge/Registros-40k-FF9800)
![Variables](https://img.shields.io/badge/Variables-25-FF9800)
![Target](https://img.shields.io/badge/Target-Action%20Taken-D32F2F)

El dataset utilizado contiene **40.000 registros** y **25 variables**. Se trata de un problema de **clasificación multiclase** en el que la variable objetivo principal obligatoria es `Action Taken`.

Targets trabajadas en el proyecto:

- **Target principal:** `Action Taken`
- **Targets adicionales opcionales:** `Severity Level`, `Attack Type`

---

## 4️⃣ Stack tecnológico

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0)
![Scikit--Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)

Tecnologías y librerías utilizadas:

- `Python`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

---

## 5️⃣ Flujo de trabajo / Metodología

![Workflow](https://img.shields.io/badge/Workflow-End%20to%20End-2E7D32)
![Train/Test](https://img.shields.io/badge/Train%2FTest-80%2F20-6D4C41)
![Evaluation](https://img.shields.io/badge/Evaluation-Multimetric-5E35B1)

El proyecto sigue un pipeline completo y coherente de clasificación:

```text
EDA -> seleccion de variables -> preprocesamiento -> train/test split
-> entrenamiento -> evaluacion -> interpretacion -> experimentos adicionales
```

Fases principales del trabajo:

1. Análisis exploratorio del dataset.
2. Revisión y selección de variables útiles para modelado tabular.
3. Construcción del preprocesamiento para variables numéricas y categóricas.
4. División estratificada en entrenamiento y prueba.
5. Entrenamiento de varios modelos base y variantes ajustadas.
6. Evaluación con métricas de clasificación y validación cruzada.
7. Interpretación de resultados y lectura metodológica final.
8. Ampliación comparativa con `Severity Level` y `Attack Type`.

---

## 6️⃣ Análisis exploratorio de datos

![EDA](https://img.shields.io/badge/EDA-Completed-00897B)
![Statistics](https://img.shields.io/badge/Statistics-Descriptive-3949AB)
![Visualization](https://img.shields.io/badge/Visualization-Included-E53935)

En la fase de análisis exploratorio se revisaron:

- forma del dataset (`shape`)
- primeras filas (`head`)
- estructura general (`info`)
- tipos de datos
- estadísticas descriptivas (`describe`)
- correlaciones entre variables numéricas
- detección simple de outliers
- valores nulos por columna
- cardinalidad de variables categóricas
- distribución de la target principal
- relación entre variables categóricas seleccionadas y `Action Taken`

Este análisis permitió detectar una señal predictiva limitada pero existente, así como justificar una selección prudente de variables antes del modelado.

### Evidencias visuales del EDA

![Matriz de correlacion](00_matriz_correlacion.png)

*Matriz de correlación entre variables numéricas.*

![Boxplots de outliers](00_boxplots_outliers.png)

*Detección visual de valores extremos mediante boxplots.*

![Valores nulos](00_valores_nulos.png)

*Distribución de valores nulos por variable.*

![Distribucion de la target](01_distribucion_target.png)

*Distribución de `Action Taken` en frecuencia y proporción.*

![Categoricas vs target](01b_categoricas_vs_target.png)

*Relación relativa entre variables categóricas seleccionadas y la target principal.*

---

## 7️⃣ Preprocesamiento

![Preprocessing](https://img.shields.io/badge/Preprocessing-Completed-F9A825)
![Imputation](https://img.shields.io/badge/Imputation-Median%20%2F%20Mode-6D4C41)
![One--Hot](https://img.shields.io/badge/Encoding-One--Hot-00838F)
![Stratified%20Split](https://img.shields.io/badge/Split-Stratified-8E24AA)

El preprocesamiento se diseñó para mantener un enfoque claro y adecuado para modelado tabular, evitando complejidad innecesaria y preservando la trazabilidad metodológica.

### Variables eliminadas antes del modelado

Se descartaron las siguientes columnas por no ser adecuadas para este enfoque tabular, por actuar como identificadores, por contener texto libre o por su dificultad para aportar valor directo sin un procesamiento adicional:

- `Timestamp`
- `Source IP Address`
- `Destination IP Address`
- `Payload Data`
- `User Information`
- `Device Information`
- `Geo-location Data`
- `Firewall Logs`
- `IDS/IPS Alerts`
- `Proxy Information`

### Revisión de variables categóricas

Tras revisar las variables candidatas, se descartaron por baja capacidad informativa o comportamiento casi constante:

- `Malware Indicators`
- `Alerts/Warnings`

### Variables finales utilizadas

- `Source Port`
- `Destination Port`
- `Protocol`
- `Packet Length`
- `Packet Type`
- `Traffic Type`
- `Anomaly Scores`
- `Attack Signature`
- `Network Segment`
- `Log Source`

### Transformaciones aplicadas

- Separación entre variables numéricas y categóricas.
- Imputación de nulos con **mediana** en variables numéricas.
- Imputación de nulos con **moda** en variables categóricas.
- Codificación **One-Hot Encoding** para variables categóricas.
- División **train/test con estratificación** para preservar la distribución de clases.

---

## 8️⃣ Modelos entrenados

![DummyClassifier](https://img.shields.io/badge/Baseline-DummyClassifier-757575)
![Decision%20Tree](https://img.shields.io/badge/Model-Decision%20Tree-43A047)
![Random%20Forest](https://img.shields.io/badge/Model-Random%20Forest-2E7D32)
![Gradient%20Boosting](https://img.shields.io/badge/Model-Gradient%20Boosting-1565C0)

Modelos comparados en la práctica:

- `Baseline (DummyClassifier)`
- `Decision Tree`
- `Decision Tree Balanced`
- `Random Forest`
- `Random Forest Balanced`
- `Random Forest Ajustado`
- `Gradient Boosting`
- `Gradient Boosting Ajustado`

El enfoque combina un **baseline sencillo** con modelos de árbol y boosting. También se probaron:

- variantes con `class_weight='balanced'`
- búsquedas acotadas de hiperparámetros para `Random Forest` y `Gradient Boosting`

Esto permite comparar modelos simples, versiones balanceadas y alternativas ajustadas dentro de una práctica académica realista.

---

## 9️⃣ Resultados principales

![Action%20Taken](https://img.shields.io/badge/Target-Action%20Taken-C62828)
![Best%20Model](https://img.shields.io/badge/Best%20Model-Gradient%20Boosting%20Ajustado-1565C0)
![F1%20Macro](https://img.shields.io/badge/F1%20Macro-0.3372-6A1B9A)
![Evaluation](https://img.shields.io/badge/Evaluation-Test%20Set-455A64)

### Resultado para la target principal `Action Taken`

> **Mejor modelo en test:** `Gradient Boosting Ajustado`  
> **Accuracy:** `0.3410`  
> **Precision macro:** `0.3395`  
> **Recall macro:** `0.3403`  
> **F1 macro:** `0.3372`

El mejor rendimiento en test dentro de esta comparación se obtuvo con **Gradient Boosting Ajustado**, aunque la lectura del proyecto debe hacerse desde una perspectiva prudente: las métricas muestran cierta capacidad predictiva, pero no una separación fuerte entre clases.

### Evidencias visuales de evaluación

![Comparacion de modelos](02_comparacion_modelos.png)

*Comparación de métricas entre los modelos entrenados.*

![Matrices de confusion](03_matrices_confusion.png)

*Matrices de confusión para los modelos evaluados.*

---

## 🔟 Interpretación de resultados

![Interpretation](https://img.shields.io/badge/Interpretation-Prudent-546E7A)
![Moderate%20Performance](https://img.shields.io/badge/Performance-Moderate-F9A825)
![Feature%20Importance](https://img.shields.io/badge/Random%20Forest-Feature%20Importance-2E7D32)

La lectura global de resultados apunta a un escenario metodológicamente correcto, pero con un rendimiento moderado. Esto es coherente con lo observado en el análisis exploratorio y con la naturaleza de las variables finalmente retenidas.

Puntos clave de interpretación:

- El problema contiene **cierta señal predictiva**, ya que los modelos mejoran respecto al baseline.
- El rendimiento sigue siendo **moderado**, lo que sugiere que la separación entre clases es solo parcial.
- El EDA ya mostraba **relaciones débiles o moderadas** entre varias variables categóricas y `Action Taken`.
- Parte de la información potencialmente útil quedó fuera por **alta cardinalidad**, **texto libre** o **baja capacidad informativa**.
- La práctica debe valorarse por la **coherencia del proceso**, la comparación entre modelos y la **calidad de la interpretación**, no solo por métricas elevadas.
- La importancia de variables extraída desde `Random Forest` debe entenderse como una **lectura complementaria** del comportamiento interno de ese modelo.
- Estas importancias **no implican causalidad** ni constituyen una jerarquía absoluta válida para cualquier algoritmo.

En conjunto, el resultado no debe presentarse como un caso de alto rendimiento predictivo, sino como una práctica sólida, bien ejecutada y bien interpretada.

![Importancia de variables](04_importancia_caracteristicas.png)

*Importancia relativa de variables en la mejor variante de `Random Forest`.*

---

## 1️⃣1️⃣ Experimentos adicionales

![Severity%20Level](https://img.shields.io/badge/Target-Severity%20Level-8E24AA)
![Attack%20Type](https://img.shields.io/badge/Target-Attack%20Type-3949AB)
![Comparative](https://img.shields.io/badge/Analysis-Comparative-546E7A)

Como ampliación comparativa opcional, también se evaluaron las targets `Severity Level` y `Attack Type` utilizando el mismo enfoque general de modelado.

### Resultados obtenidos

- **Severity Level**
  - Mejor modelo: `Random Forest`
  - Mejor F1 macro: `0.3349`

- **Attack Type**
  - Mejor modelo: `Random Forest`
  - Mejor F1 macro: `0.3373`

### Lectura de esta ampliación

- Las dos targets opcionales quedan **muy cerca** del rendimiento obtenido en `Action Taken`.
- Esto sugiere que, con las variables retenidas, la dificultad de los tres problemas es comparable.
- Estos experimentos se presentan como **ampliación comparativa**, no como eje central del proyecto.
- La **target principal sigue siendo `Action Taken`** y es la que estructura la práctica.

---

## 1️⃣2️⃣ Visualizaciones del proyecto

![PNG](https://img.shields.io/badge/Output-PNG%20Charts-E53935)
![Visual%20Evidence](https://img.shields.io/badge/Visual%20Evidence-Included-00897B)
![Analysis](https://img.shields.io/badge/Analysis-Supported-3949AB)

Gráficas generadas por el script y su utilidad:

- `00_matriz_correlacion.png`  
  Muestra la correlación entre variables numéricas.

- `00_boxplots_outliers.png`  
  Permite revisar visualmente posibles outliers.

- `00_valores_nulos.png`  
  Resume la presencia de valores nulos por columna.

- `01_distribucion_target.png`  
  Representa la distribución de la target principal.

- `01b_categoricas_vs_target.png`  
  Muestra la relación relativa entre variables categóricas seleccionadas y `Action Taken`.

- `02_comparacion_modelos.png`  
  Compara el rendimiento global entre modelos.

- `03_matrices_confusion.png`  
  Ayuda a entender errores y confusiones entre clases.

- `04_importancia_caracteristicas.png`  
  Ofrece una lectura complementaria sobre relevancia relativa de variables en `Random Forest`.

---

## 1️⃣3️⃣ Estructura del proyecto

![Project%20Structure](https://img.shields.io/badge/Project-Structure-6D4C41)
![GitHub](https://img.shields.io/badge/GitHub-Ready-181717?logo=github&logoColor=white)
![Portfolio](https://img.shields.io/badge/Portfolio-Friendly-0A66C2)

```bash
.
├── cybersecurity_attacks.csv
├── main.py
├── README.md
├── README.en.md
├── 00_matriz_correlacion.png
├── 00_boxplots_outliers.png
├── 00_valores_nulos.png
├── 01_distribucion_target.png
├── 01b_categoricas_vs_target.png
├── 02_comparacion_modelos.png
├── 03_matrices_confusion.png
└── 04_importancia_caracteristicas.png
```

---

## 1️⃣4️⃣ Cómo ejecutar el proyecto

![Run](https://img.shields.io/badge/Execution-Python%20Script-3776AB?logo=python&logoColor=white)
![CLI](https://img.shields.io/badge/Interface-CLI-455A64)
![Reproducible](https://img.shields.io/badge/Setup-Reproducible-2E7D32)

### 1. Clonar el repositorio

```bash
git clone https://github.com/David-Navarro-Oliver/practica-machine-learning.git
cd practica-machine-learning
```

### 2. Instalar dependencias

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 3. Colocar el dataset en la raíz del proyecto

Asegúrate de que el archivo `cybersecurity_attacks.csv` esté en la carpeta principal del proyecto, junto a `main.py`.

### 4. Ejecutar el script

```bash
python main.py
```

Al ejecutar el script se generarán las gráficas y se mostrará por consola el análisis completo, el entrenamiento, la evaluación y la conclusión final.

---

## 1️⃣5️⃣ Conclusiones

![Conclusion](https://img.shields.io/badge/Conclusion-Academic-2E7D32)
![Methodology](https://img.shields.io/badge/Methodology-Coherent-1565C0)
![Interpretation](https://img.shields.io/badge/Interpretation-Honest-6A1B9A)
![Portfolio](https://img.shields.io/badge/Portfolio-Ready-181717?logo=github&logoColor=white)

Esta práctica desarrolla un flujo completo de clasificación en Python, incluyendo análisis exploratorio, selección razonada de variables, preprocesamiento, entrenamiento de modelos, evaluación y conclusión interpretada. Desde el punto de vista metodológico, el proyecto puede considerarse **sólido, coherente y defendible**.

La capacidad predictiva observada existe, pero es **limitada**. El rendimiento moderado resulta coherente con la señal disponible en el dataset y con el hecho de que las variables retenidas separan solo parcialmente las clases. En este contexto, el valor del trabajo está en la **comparación razonada de modelos**, en la **interpretación honesta de resultados** y en la **solidez del proceso completo**.

Como pieza de portfolio técnico, este proyecto refleja criterio en la preparación de datos, consistencia metodológica en la evaluación y una lectura final alineada con lo que realmente muestran los datos.

---

## 1️⃣6️⃣ Autor

![Autor](https://img.shields.io/badge/Autor-David%20Navarro-0A66C2)
![Machine%20Learning](https://img.shields.io/badge/Area-Machine%20Learning-1565C0)
![GitHub](https://img.shields.io/badge/GitHub-Portfolio-181717?logo=github&logoColor=white)

**David Navarro**  
Proyecto académico realizado en KeepCoding para el módulo de IA y ciberseguridad.
