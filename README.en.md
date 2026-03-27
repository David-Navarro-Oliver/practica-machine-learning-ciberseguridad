# Machine Learning Practice - Cyber Attack Classification

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification-0A66C2)
![Cybersecurity](https://img.shields.io/badge/Domain-Cybersecurity-37474F)
![Portfolio](https://img.shields.io/badge/GitHub-Portfolio-181717?logo=github&logoColor=white)

[🇪🇸 Versión en español](README.md) · [🇬🇧 English version](README.en.md)

*Multiclass classification project in Python focused on predicting the action taken in response to cyber attacks using the `cybersecurity_attacks.csv` dataset, with an academic approach and a GitHub portfolio-ready presentation.*

---

## 1️⃣ Project overview

![Classification](https://img.shields.io/badge/Task-Classification-1565C0)
![Multiclass](https://img.shields.io/badge/Problem-Multiclass-6A1B9A)
![Action%20Taken](https://img.shields.io/badge/Main%20Target-Action%20Taken-C62828)

This project addresses a multiclass classification problem based on the `cybersecurity_attacks.csv` dataset, with the main goal of predicting the mandatory target variable `Action Taken`. The work follows a complete workflow including exploratory data analysis, feature selection, preprocessing, model training, evaluation, and final interpretation. In addition, `Severity Level` and `Attack Type` are included as optional comparative targets without displacing the main focus of the project.

### Key results

![Best Model](https://img.shields.io/badge/Best%20Model-Gradient%20Boosting%20Ajustado-1565C0)
![Accuracy](https://img.shields.io/badge/Accuracy-0.3410-2E7D32)
![F1 Macro](https://img.shields.io/badge/F1%20Macro-0.3372-6A1B9A)
![Dataset](https://img.shields.io/badge/Dataset-40,000%20rows%20%7C%2025%20features-546E7A)

- **Main target:** `Action Taken`
- **Best model:** `Gradient Boosting Ajustado`
- **Accuracy:** `0.3410`
- **Macro F1:** `0.3372`
- **Dataset rows:** `40,000`
- **Features:** `25`

---

## 2️⃣ Objectives

![EDA](https://img.shields.io/badge/EDA-Included-00897B)
![Preprocessing](https://img.shields.io/badge/Preprocessing-Included-F9A825)
![Evaluation](https://img.shields.io/badge/Model%20Evaluation-Included-5E35B1)

- Analyze the dataset structure and quality.
- Prepare the data for tabular modeling.
- Train several classification models.
- Compare results across different approaches.
- Interpret performance with methodological rigor.
- Extend the analysis with additional comparative targets.

---

## 3️⃣ Dataset

![Dataset](https://img.shields.io/badge/Dataset-cybersecurity__attacks.csv-546E7A)
![Rows](https://img.shields.io/badge/Rows-40k-FF9800)
![Features](https://img.shields.io/badge/Features-25-FF9800)
![Target](https://img.shields.io/badge/Target-Action%20Taken-D32F2F)

The dataset contains **40,000 rows** and **25 features**. It is a **multiclass classification** problem whose mandatory main target is `Action Taken`.

Targets used in the project:

- **Main target:** `Action Taken`
- **Optional additional targets:** `Severity Level`, `Attack Type`

---

## 4️⃣ Tech stack

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0)
![Scikit--Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)

Technologies and libraries used:

- `Python`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

---

## 5️⃣ Workflow / Methodology

![Workflow](https://img.shields.io/badge/Workflow-End%20to%20End-2E7D32)
![Train/Test](https://img.shields.io/badge/Train%2FTest-80%2F20-6D4C41)
![Evaluation](https://img.shields.io/badge/Evaluation-Multimetric-5E35B1)

The project follows a complete and coherent classification pipeline:

```text
EDA -> feature selection -> preprocessing -> train/test split
-> training -> evaluation -> interpretation -> additional experiments
```

Main stages of the workflow:

1. Exploratory data analysis.
2. Review and selection of variables suitable for tabular modeling.
3. Preprocessing design for numerical and categorical variables.
4. Stratified train/test split.
5. Training of baseline and adjusted models.
6. Evaluation with classification metrics and cross-validation.
7. Interpretation of results and final methodological reading.
8. Comparative extension with `Severity Level` and `Attack Type`.

---

## 6️⃣ Exploratory data analysis

![EDA](https://img.shields.io/badge/EDA-Completed-00897B)
![Statistics](https://img.shields.io/badge/Statistics-Descriptive-3949AB)
![Visualization](https://img.shields.io/badge/Visualization-Included-E53935)

The exploratory analysis reviewed:

- dataset shape
- first rows (`head`)
- general structure (`info`)
- data types
- descriptive statistics (`describe`)
- correlations among numerical variables
- simple outlier detection
- missing values by column
- categorical cardinality
- distribution of the main target
- relationship between selected categorical variables and `Action Taken`

This stage revealed a limited but existing predictive signal and supported a cautious feature selection before modeling.

### EDA visual evidence

![Correlation matrix](00_matriz_correlacion.png)

*Correlation matrix for numerical variables.*

![Outlier boxplots](00_boxplots_outliers.png)

*Visual inspection of potential outliers through boxplots.*

![Missing values](00_valores_nulos.png)

*Distribution of missing values by variable.*

![Target distribution](01_distribucion_target.png)

*Distribution of `Action Taken` in counts and proportions.*

![Categoricals vs target](01b_categoricas_vs_target.png)

*Relative relationship between selected categorical variables and the main target.*

---

## 7️⃣ Preprocessing

![Preprocessing](https://img.shields.io/badge/Preprocessing-Completed-F9A825)
![Imputation](https://img.shields.io/badge/Imputation-Median%20%2F%20Mode-6D4C41)
![One--Hot](https://img.shields.io/badge/Encoding-One--Hot-00838F)
![Stratified%20Split](https://img.shields.io/badge/Split-Stratified-8E24AA)

The preprocessing stage was designed to keep a clear and suitable tabular modeling approach, avoiding unnecessary complexity while preserving methodological traceability.

### Features removed before modeling

The following columns were excluded because they were not appropriate for this tabular approach, acted as identifiers, contained free text, or would require a more advanced treatment:

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

### Review of categorical variables

After reviewing the candidate features, the following variables were discarded due to low information value or near-constant behavior:

- `Malware Indicators`
- `Alerts/Warnings`

### Final selected features

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

### Applied transformations

- Separation of numerical and categorical variables.
- Missing value imputation with **median** for numerical variables.
- Missing value imputation with **mode** for categorical variables.
- **One-Hot Encoding** for categorical variables.
- **Stratified train/test split** to preserve class distribution.

---

## 8️⃣ Trained models

![DummyClassifier](https://img.shields.io/badge/Baseline-DummyClassifier-757575)
![Decision%20Tree](https://img.shields.io/badge/Model-Decision%20Tree-43A047)
![Random%20Forest](https://img.shields.io/badge/Model-Random%20Forest-2E7D32)
![Gradient%20Boosting](https://img.shields.io/badge/Model-Gradient%20Boosting-1565C0)

Models compared in this practice:

- `Baseline (DummyClassifier)`
- `Decision Tree`
- `Decision Tree Balanced`
- `Random Forest`
- `Random Forest Balanced`
- `Random Forest Ajustado`
- `Gradient Boosting`
- `Gradient Boosting Ajustado`

The project compares a **simple baseline** against tree-based and boosting models. It also includes:

- variants with `class_weight='balanced'`
- limited hyperparameter searches for `Random Forest` and `Gradient Boosting`

This makes it possible to compare simple models, balanced variants, and tuned alternatives within a realistic academic workflow.

---

## 9️⃣ Main results

![Action%20Taken](https://img.shields.io/badge/Target-Action%20Taken-C62828)
![Best%20Model](https://img.shields.io/badge/Best%20Model-Gradient%20Boosting%20Ajustado-1565C0)
![F1%20Macro](https://img.shields.io/badge/F1%20Macro-0.3372-6A1B9A)
![Evaluation](https://img.shields.io/badge/Evaluation-Test%20Set-455A64)

### Result for the main target `Action Taken`

> **Best test model:** `Gradient Boosting Ajustado`  
> **Accuracy:** `0.3410`  
> **Macro precision:** `0.3395`  
> **Macro recall:** `0.3403`  
> **Macro F1:** `0.3372`

The best performance on the test set within this comparison was achieved by **Gradient Boosting Ajustado**, although the project should be read from a prudent perspective: the metrics suggest some predictive capacity, but not a strong class separation.

### Evaluation visual evidence

![Model comparison](02_comparacion_modelos.png)

*Metric comparison across the trained models.*

![Confusion matrices](03_matrices_confusion.png)

*Confusion matrices for the evaluated models.*

---

## 🔟 Results interpretation

![Interpretation](https://img.shields.io/badge/Interpretation-Prudent-546E7A)
![Moderate%20Performance](https://img.shields.io/badge/Performance-Moderate-F9A825)
![Feature%20Importance](https://img.shields.io/badge/Random%20Forest-Feature%20Importance-2E7D32)

The overall reading of the results points to a methodologically correct scenario, but with moderate performance. This is consistent with what was observed during exploratory analysis and with the nature of the features retained for modeling.

Key interpretation points:

- The problem contains **some predictive signal**, since the models improve over the baseline.
- Performance remains **moderate**, which suggests that class separation is only partial.
- The EDA already showed **weak to moderate relationships** between several categorical variables and `Action Taken`.
- Some potentially useful information was left out due to **high cardinality**, **free text**, or **low information value**.
- The project should be valued for the **coherence of the process**, the model comparison, and the **quality of the interpretation**, not only for high metrics.
- Feature importance from `Random Forest` should be understood as a **complementary reading** of that model's internal behavior.
- These importance values **do not imply causality** and are not an absolute ranking valid for every model.

Overall, this should not be presented as a high-performance predictive case, but as a solid, well-executed, and well-interpreted practice.

![Feature importance](04_importancia_caracteristicas.png)

*Relative feature importance from the best `Random Forest` variant.*

---

## 1️⃣1️⃣ Additional experiments

![Severity%20Level](https://img.shields.io/badge/Target-Severity%20Level-8E24AA)
![Attack%20Type](https://img.shields.io/badge/Target-Attack%20Type-3949AB)
![Comparative](https://img.shields.io/badge/Analysis-Comparative-546E7A)

As an optional comparative extension, the project also evaluates the targets `Severity Level` and `Attack Type` using the same general modeling approach.

### Obtained results

- **Severity Level**
  - Best model: `Random Forest`
  - Best macro F1: `0.3349`

- **Attack Type**
  - Best model: `Random Forest`
  - Best macro F1: `0.3373`

### Reading of this extension

- Both optional targets remain **very close** to the performance obtained for `Action Taken`.
- This suggests that, with the retained variables, the three problems have a comparable level of difficulty.
- These experiments are presented as a **comparative extension**, not as the core of the project.
- The **main target remains `Action Taken`** and continues to organize the overall practice.

---

## 1️⃣2️⃣ Project visualizations

![PNG](https://img.shields.io/badge/Output-PNG%20Charts-E53935)
![Visual%20Evidence](https://img.shields.io/badge/Visual%20Evidence-Included-00897B)
![Analysis](https://img.shields.io/badge/Analysis-Supported-3949AB)

Charts generated by the script and their purpose:

- `00_matriz_correlacion.png`  
  Shows correlations among numerical variables.

- `00_boxplots_outliers.png`  
  Supports the visual inspection of potential outliers.

- `00_valores_nulos.png`  
  Summarizes missing values by column.

- `01_distribucion_target.png`  
  Displays the distribution of the main target.

- `01b_categoricas_vs_target.png`  
  Shows the relative relationship between selected categorical variables and `Action Taken`.

- `02_comparacion_modelos.png`  
  Compares overall model performance.

- `03_matrices_confusion.png`  
  Helps understand classification errors and class confusions.

- `04_importancia_caracteristicas.png`  
  Provides a complementary reading of relative feature relevance in `Random Forest`.

---

## 1️⃣3️⃣ Project structure

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

## 1️⃣4️⃣ How to run the project

![Run](https://img.shields.io/badge/Execution-Python%20Script-3776AB?logo=python&logoColor=white)
![CLI](https://img.shields.io/badge/Interface-CLI-455A64)
![Reproducible](https://img.shields.io/badge/Setup-Reproducible-2E7D32)

### 1. Clone the repository

```bash
git clone https://github.com/David-Navarro-Oliver/practica-machine-learning.git
cd practica-machine-learning
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 3. Place the dataset in the project root

Make sure the file `cybersecurity_attacks.csv` is located in the project root, next to `main.py`.

### 4. Run the script

```bash
python main.py
```

Running the script will generate the figures and display the full analysis, training, evaluation, and final interpretation in the console.

---

## 1️⃣5️⃣ Conclusions

![Conclusion](https://img.shields.io/badge/Conclusion-Academic-2E7D32)
![Methodology](https://img.shields.io/badge/Methodology-Coherent-1565C0)
![Interpretation](https://img.shields.io/badge/Interpretation-Honest-6A1B9A)
![Portfolio](https://img.shields.io/badge/Portfolio-Ready-181717?logo=github&logoColor=white)

This project develops a complete classification workflow in Python, including exploratory data analysis, feature selection, preprocessing, model training, evaluation, and an interpreted conclusion. From a methodological perspective, it can be considered **solid, coherent, and academically defensible**.

There is some predictive capacity, but it remains **limited**. The moderate performance is consistent with the signal available in the dataset and with the fact that the retained variables only partially separate the classes. In this context, the project's value lies in the **reasoned comparison of models**, the **honest interpretation of results**, and the **overall solidity of the process**.

As a technical portfolio piece, this project reflects judgment in data preparation, methodological consistency in evaluation, and a final reading aligned with what the data actually supports.

---

## 1️⃣6️⃣ Author

![Author](https://img.shields.io/badge/Author-David%20Navarro-0A66C2)
![Machine%20Learning](https://img.shields.io/badge/Area-Machine%20Learning-1565C0)
![GitHub](https://img.shields.io/badge/GitHub-Portfolio-181717?logo=github&logoColor=white)

**David Navarro**  
Academic Machine Learning project oriented to a technical GitHub portfolio.
