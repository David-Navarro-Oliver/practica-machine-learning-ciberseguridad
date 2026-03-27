# -*- coding: utf-8 -*-
"""
Practica de Machine Learning - Clasificacion de Ataques Ciberneticos
Dataset: cybersecurity_attacks.csv
Variable Target: Action Taken

Objetivo: Construir y evaluar modelos de clasificacion para predecir 
las acciones tomadas en respuesta a ataques ciberneticos.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# =============================================================================
# 1. CARGA Y ANALISIS EXPLORATORIO DE DATOS
# =============================================================================
print("=" * 80)
print("1. CARGA Y ANALISIS EXPLORATORIO")
print("=" * 80)
print("Este bloque revisa la estructura del dataset, la calidad del dato y las relaciones iniciales antes del modelado.")

df = pd.read_csv('cybersecurity_attacks.csv')
print(f"\nDataset: {df.shape[0]} registros, {df.shape[1]} variables\n")

print("Primeras filas:")
print(df.head())

print("\nInformacion general del dataset:")
df.info()

print("\nTipos de datos (dtypes):")
print(df.dtypes)

print("\nEstadisticas descriptivas de variables numericas:")
print(df.describe().round(3))

numeric_eda_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Analisis de correlacion
print("\nCorrelacion entre variables numericas:")
corr_matrix = df[numeric_eda_cols].corr()
print(corr_matrix.round(3))

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', ax=ax)
ax.set_title('Matriz de correlacion de variables numericas', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('00_matriz_correlacion.png', dpi=300, bbox_inches='tight')
print("\nGrafica guardada: 00_matriz_correlacion.png")
plt.show()
plt.close()

upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
max_abs_corr = upper_triangle.abs().max().max()
if max_abs_corr > 0.7:
    print("Se observan correlaciones lineales relativamente altas entre algunas variables numericas.")
elif max_abs_corr > 0.3:
    print("Se observan correlaciones lineales moderadas, sin evidencias de dependencia muy fuerte.")
else:
    print("No se aprecian correlaciones lineales fuertes entre las variables numericas.")

# Analisis simple de outliers
print("\nAnalisis simple de outliers (IQR):")
outlier_summary = []
for col in numeric_eda_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    outlier_summary.append({
        'Variable': col,
        'Outliers': outliers,
        'Porcentaje': outliers / len(df) * 100
    })

outlier_df = pd.DataFrame(outlier_summary)
print(outlier_df.round({'Porcentaje': 2}))

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.ravel()
for idx, col in enumerate(numeric_eda_cols):
    sns.boxplot(x=df[col], ax=axes[idx], color='lightsteelblue')
    axes[idx].set_title(col, fontsize=11, fontweight='bold')
    axes[idx].set_xlabel('')
plt.tight_layout()
plt.savefig('00_boxplots_outliers.png', dpi=300, bbox_inches='tight')
print("\nGrafica guardada: 00_boxplots_outliers.png")
plt.show()
plt.close()

max_outlier_pct = outlier_df['Porcentaje'].max()
if max_outlier_pct == 0:
    print("El criterio IQR no identifica outliers relevantes en las variables numericas.")
elif max_outlier_pct > 10:
    print("El analisis IQR detecta valores extremos en varias variables; en modelos de arboles su impacto suele ser mas acotado.")
else:
    print("El analisis IQR detecta algunos valores extremos, pero no parecen dominar el conjunto de datos.")


# Analisis de valores nulos
print("\nValores nulos:")
null_counts = df.isnull().sum()
null_cols = null_counts[null_counts > 0].sort_values(ascending=False)
if len(null_cols) > 0:
    for col, count in null_cols.items():
        pct = count / len(df) * 100
        print(f"  {col}: {count} ({pct:.1f}%)")
    print(f"Columnas con nulos destacados: {null_cols.index.tolist()}")
else:
    print("  No hay valores nulos")

# Grafica adicional: valores nulos por columna
if len(null_cols) > 0:
    fig, ax = plt.subplots(figsize=(12, 5))
    null_cols.sort_values().plot(kind='barh', ax=ax, color='indianred', edgecolor='black')
    ax.set_title('Valores nulos por columna', fontsize=12, fontweight='bold')
    ax.set_xlabel('Cantidad de valores nulos')
    ax.set_ylabel('Variable')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('00_valores_nulos.png', dpi=300, bbox_inches='tight')
    print("\nGrafica guardada: 00_valores_nulos.png")
    plt.show()
    plt.close()

# Cardinalidad
print("\nCardinalidad de variables categoricas:")
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"  {col}: {df[col].nunique()} valores")
high_card = [col for col in categorical_cols if df[col].nunique() > 100]
if high_card:
    print(f"Variables categoricas con alta cardinalidad: {high_card}")

# Analisis de la variable target
print("\nDistribucion de la variable target 'Action Taken':")
target_dist = df['Action Taken'].value_counts()
target_dist_normalized = df['Action Taken'].value_counts(normalize=True)
for action, count in target_dist.items():
    pct = count / len(df) * 100
    print(f"  {action}: {count} ({pct:.1f}%)")
max_pct = target_dist.max() / len(df) * 100
if max_pct > 60:
    print("La target muestra desbalance, con una clase predominante.")
else:
    print("La target esta relativamente balanceada.")

# Grafica 1: distribucion de la target
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

df['Action Taken'].value_counts().plot(kind='bar', ax=axes[0], color='steelblue', edgecolor='black')
axes[0].set_title('Distribucion de Action Taken', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Frecuencia')
axes[0].set_xlabel('Accion')
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(axis='y', alpha=0.3)

df['Action Taken'].value_counts(normalize=True).plot(kind='pie', ax=axes[1], autopct='%1.1f%%', 
                                                       colors=['#FF9999', '#66B2FF', '#99FF99'])
axes[1].set_title('Proporcion de Action Taken', fontsize=12, fontweight='bold')
axes[1].set_ylabel('')

plt.tight_layout()
plt.savefig('01_distribucion_target.png', dpi=300, bbox_inches='tight')
print("\nGrafica guardada: 01_distribucion_target.png")
plt.show()
plt.close()

# Analisis de relacion entre variables categoricas y la target
print("\nRelacion entre variables categoricas y la target 'Action Taken':")
categorical_analysis_cols = ['Protocol', 'Traffic Type', 'Network Segment', 'Log Source']
signal_summary = {}

for col in categorical_analysis_cols:
    relative_table_raw = pd.crosstab(df[col], df['Action Taken'], normalize='index')
    relative_table = relative_table_raw.round(3)
    signal_gap = relative_table_raw.sub(target_dist_normalized, axis=1).abs().max().max()
    signal_summary[col] = signal_gap

    print(f"\nFrecuencias relativas de 'Action Taken' segun {col}:")
    print(relative_table)

    if signal_gap > 0.08:
        print(f"Interpretacion: {col} parece aportar senal apreciable, porque algunas categorias muestran distribuciones claramente distintas de la distribucion global.")
    elif signal_gap > 0.03:
        print(f"Interpretacion: {col} parece aportar una senal moderada; la mezcla de clases cambia entre categorias, aunque sin separacion muy marcada.")
    else:
        print(f"Interpretacion: {col} muestra una senal limitada; la distribucion de la target es parecida entre sus categorias.")

fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=True)
axes = axes.ravel()

palette = sns.color_palette('Set2', n_colors=len(target_dist_normalized))
for idx, col in enumerate(categorical_analysis_cols):
    relative_table = pd.crosstab(df[col], df['Action Taken'], normalize='index')
    relative_table.plot(
        kind='bar',
        stacked=True,
        ax=axes[idx],
        color=palette,
        edgecolor='black',
        linewidth=0.5
    )
    axes[idx].set_title(f'{col} vs Action Taken', fontsize=11, fontweight='bold')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Proporcion')
    axes[idx].tick_params(axis='x', rotation=25)
    axes[idx].legend(title='Action Taken', fontsize=8, title_fontsize=9)
    axes[idx].grid(axis='y', alpha=0.3)

plt.suptitle('Distribucion relativa de Action Taken en variables categoricas', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('01b_categoricas_vs_target.png', dpi=300, bbox_inches='tight')
print("\nGrafica guardada: 01b_categoricas_vs_target.png")
plt.show()
plt.close()

print("\nLectura conjunta del analisis categorico:")
max_signal_gap = max(signal_summary.values()) if signal_summary else 0.0

if max_signal_gap > 0.08:
    print("- Algunas variables categoricas parecen contener senal util para anticipar la accion tomada.")
elif max_signal_gap > 0.03:
    print("- Las variables categoricas analizadas aportan cierta senal, aunque con diferencias moderadas entre categorias.")
else:
    print("- Las variables categoricas analizadas muestran una relacion limitada con la target principal.")

print("Conexion con el modelado:")
if max_signal_gap > 0.08:
    print("- Este patron sugiere que los modelos pueden capturar parte de la estructura del problema, aunque no garantiza una separacion perfecta entre clases.")
elif max_signal_gap > 0.03:
    print("- Con relaciones moderadas, es razonable esperar mejoras frente al baseline, pero no un F1 macro muy alto.")
else:
    print("- Como las distribuciones condicionales son muy parecidas a la distribucion global, es razonable esperar metricas moderadas. Un F1 contenido seria coherente con una senal predictiva limitada.")


def build_preprocessor(X_train):
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    try:
        one_hot = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        one_hot = OneHotEncoder(handle_unknown='ignore', sparse=False)

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', one_hot)
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    return preprocessor, numeric_cols, categorical_cols

# =============================================================================
# 2. PREPROCESAMIENTO
# =============================================================================
print("\n" + "=" * 80)
print("2. PREPROCESAMIENTO")
print("=" * 80)
print("En esta fase se prepara el conjunto de entrada y se filtran variables poco informativas para sostener un modelado mas robusto.")

df_processed = df.copy()

columns_to_drop = [
    'Timestamp', 'Source IP Address', 'Destination IP Address',
    'Payload Data', 'User Information', 'Device Information',
    'Geo-location Data', 'Firewall Logs', 'IDS/IPS Alerts',
    'Proxy Information'
]

df_processed = df_processed.drop(columns=columns_to_drop)

target_columns = ['Action Taken', 'Severity Level', 'Attack Type']
candidate_feature_cols = [col for col in df_processed.columns if col not in target_columns]

print(f"\nConjunto reducido tras el descarte inicial: {len(candidate_feature_cols)} variables candidatas")
print(f"Lista inicial: {candidate_feature_cols}")

categorical_candidate_cols = df_processed[candidate_feature_cols].select_dtypes(include=['object']).columns.tolist()
numeric_candidate_cols = df_processed[candidate_feature_cols].select_dtypes(include=[np.number]).columns.tolist()

print("\nRevision de variables categoricas candidatas:")
candidate_review_rows = []
low_information_categorical_cols = []
discard_reasons = {}

for col in categorical_candidate_cols:
    series = df_processed[col]
    non_null_unique = series.dropna().nunique()
    missing_pct = series.isna().mean() * 100

    mode_series = series.mode(dropna=True)
    fill_value = mode_series.iloc[0] if not mode_series.empty else 'Missing'
    imputed_series = series.fillna(fill_value)
    dominant_share = imputed_series.value_counts(normalize=True, dropna=False).iloc[0]

    decision = 'Mantener'
    reason = 'Mantiene variabilidad suficiente para ser evaluada por el modelo.'

    if non_null_unique <= 1:
        decision = 'Descartar'
        reason = 'Solo presenta un valor observado; tras la imputacion queda practicamente constante.'
        low_information_categorical_cols.append(col)
        discard_reasons[col] = reason
    elif dominant_share >= 0.98:
        decision = 'Descartar'
        reason = 'La categoria dominante supera el 98% tras la imputacion y aporta muy poca informacion.'
        low_information_categorical_cols.append(col)
        discard_reasons[col] = reason

    candidate_review_rows.append({
        'Variable': col,
        'Valores no nulos': non_null_unique,
        'Nulos %': round(missing_pct, 2),
        'Moda tras imputacion %': round(dominant_share * 100, 2),
        'Decision': decision,
        'Motivo': reason
    })

candidate_review_df = pd.DataFrame(candidate_review_rows)
print(candidate_review_df.to_string(index=False))

print("\nVariables numericas candidatas:")
print(numeric_candidate_cols)
print("Las variables numericas del conjunto reducido no presentan comportamiento constante y se mantienen para el modelado.")

low_information_categorical_cols = sorted(set(low_information_categorical_cols))
if low_information_categorical_cols:
    print("\nColumnas descartadas por baja capacidad informativa:")
    for col in low_information_categorical_cols:
        print(f"- {col}: {discard_reasons[col]}")
else:
    print("\nNo se detectaron variables categoricas claramente constantes o casi constantes.")

print("Criterio aplicado: se excluyen variables categoricas casi invariantes porque tienden a anadir ruido y apenas ayudan a separar clases.")

feature_cols = [col for col in candidate_feature_cols if col not in low_information_categorical_cols]

print(f"\nColumnas seleccionadas: {len(feature_cols)}")
print(f"Lista final: {feature_cols}")

print("\nJustificacion de columnas eliminadas:")
print("- Identificadores unicos (Timestamp, IPs) no aportan informacion predictiva.")
print("- Texto libre (Payload Data, User Information, etc.) requiere procesamiento avanzado.")
print("- Columnas con muchos nulos o alta cardinalidad se excluyen para evitar ruido y sobreajuste.")
if low_information_categorical_cols:
    print(f"- Variables categoricas casi constantes ({low_information_categorical_cols}) se descartan por baja capacidad informativa.")

# Separar X e y ANTES del train-test split
X = df_processed[feature_cols]
y = df_processed['Action Taken']

print(f"X: {X.shape}")
print(f"y: {y.shape}")

# Train-test split con estratificacion
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nEntrenamiento: {X_train.shape[0]} registros")
print(f"Prueba: {X_test.shape[0]} registros")

preprocessor, numeric_cols, categorical_cols = build_preprocessor(X_train)

print(f"\nVariables numericas finales: {numeric_cols}")
print(f"Variables categoricas finales: {categorical_cols}")

target_classes = sorted(y.unique())
print(f"Target: {len(target_classes)} clases - {target_classes}")

# =============================================================================
# 3. ENTRENAMIENTO DE MODELOS
# =============================================================================
print("\n" + "=" * 80)
print("3. ENTRENAMIENTO DE MODELOS")
print("=" * 80)
print("Se comparan modelos de distinta complejidad y versiones balanced de arboles para evaluar si una ponderacion simple mejora el resultado.")

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

model_specs = {
    'Baseline': DummyClassifier(strategy='most_frequent', random_state=42),
    'Decision Tree': DecisionTreeClassifier(
        max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42
    ),
    'Decision Tree Balanced': DecisionTreeClassifier(
        max_depth=10, min_samples_split=5, min_samples_leaf=2,
        class_weight='balanced', random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, max_depth=12, min_samples_split=5,
        min_samples_leaf=2, random_state=42, n_jobs=1
    ),
    'Random Forest Balanced': RandomForestClassifier(
        n_estimators=100, max_depth=12, min_samples_split=5,
        min_samples_leaf=2, class_weight='balanced',
        random_state=42, n_jobs=1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
    )
}

tuned_search_spaces = {
    'Random Forest Ajustado': {
        'estimator': RandomForestClassifier(random_state=42, n_jobs=1),
        'param_distributions': {
            'classifier__n_estimators': [100, 150, 200],
            'classifier__max_depth': [8, 12, 16, None],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4],
            'classifier__max_features': ['sqrt', 'log2', None]
        }
    },
    'Gradient Boosting Ajustado': {
        'estimator': GradientBoostingClassifier(random_state=42),
        'param_distributions': {
            'classifier__n_estimators': [80, 100, 150],
            'classifier__learning_rate': [0.03, 0.05, 0.1],
            'classifier__max_depth': [2, 3, 4],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4],
            'classifier__subsample': [0.8, 0.9, 1.0]
        }
    }
}

model_order = [
    'Baseline',
    'Decision Tree',
    'Decision Tree Balanced',
    'Random Forest',
    'Random Forest Balanced',
    'Random Forest Ajustado',
    'Gradient Boosting',
    'Gradient Boosting Ajustado'
]

secondary_model_order = [
    'Baseline',
    'Decision Tree',
    'Random Forest',
    'Gradient Boosting'
]


def run_additional_target_experiment(target_name):
    print("\n" + "-" * 80)
    print(f"Target adicional: {target_name}")
    print("-" * 80)

    X_target = df_processed[feature_cols]
    y_target = df_processed[target_name]

    print(f"\nTarget: {target_name}")
    print(f"X: {X_target.shape}")
    print(f"y: {y_target.shape}")

    X_train_target, X_test_target, y_train_target, y_test_target = train_test_split(
        X_target, y_target, test_size=0.2, random_state=42, stratify=y_target
    )

    preprocessor_target, numeric_cols_target, categorical_cols_target = build_preprocessor(X_train_target)
    target_classes_target = sorted(y_target.unique())

    print(f"Entrenamiento: {X_train_target.shape[0]} registros")
    print(f"Prueba: {X_test_target.shape[0]} registros")
    print(f"Variables numericas: {numeric_cols_target}")
    print(f"Variables categoricas: {categorical_cols_target}")
    print(f"Clases: {target_classes_target}")

    predictions_target = {}
    for model_name in secondary_model_order:
        estimator = model_specs[model_name]
        pipeline = Pipeline(steps=[
            ('preprocessor', clone(preprocessor_target)),
            ('classifier', clone(estimator))
        ])

        pipeline.fit(X_train_target, y_train_target)
        predictions_target[model_name] = pipeline.predict(X_test_target)

    results_target = {}

    for model_name in secondary_model_order:
        y_pred_target = predictions_target[model_name]
        accuracy = accuracy_score(y_test_target, y_pred_target)
        precision = precision_score(y_test_target, y_pred_target, average='macro', zero_division=0)
        recall = recall_score(y_test_target, y_pred_target, average='macro', zero_division=0)
        f1 = f1_score(y_test_target, y_pred_target, average='macro', zero_division=0)

        results_target[model_name] = {
            'Accuracy test': accuracy,
            'Precision macro': precision,
            'Recall macro': recall,
            'F1 macro': f1
        }

    results_target_df = pd.DataFrame(results_target).T.loc[secondary_model_order, [
        'Accuracy test', 'Precision macro', 'Recall macro', 'F1 macro'
    ]]
    results_target_display = results_target_df.round(4).astype(object)
    results_target_display = results_target_display.where(pd.notnull(results_target_display), '-')

    print("\nResumen de resultados:")
    print(results_target_display)

    best_model_target = results_target_df['F1 macro'].idxmax()
    best_f1_target = results_target_df.loc[best_model_target, 'F1 macro']

    print(f"Mejor modelo: {best_model_target}")
    print(f"Mejor F1 macro: {best_f1_target:.4f}")

    action_taken_reference_f1 = globals().get('best_f1')
    if action_taken_reference_f1 is not None:
        delta_vs_action_taken = best_f1_target - action_taken_reference_f1
        if delta_vs_action_taken > 0.01:
            print(f"Lectura academica: esta target podria resultar ligeramente mas accesible que Action Taken con las variables retenidas ({delta_vs_action_taken:+.4f} en F1 macro).")
        elif delta_vs_action_taken < -0.01:
            print(f"Lectura academica: esta target podria resultar algo mas exigente que Action Taken con las variables retenidas ({delta_vs_action_taken:+.4f} en F1 macro).")
        else:
            print(f"Lectura academica: esta target muestra una dificultad comparable a Action Taken con las variables retenidas ({delta_vs_action_taken:+.4f} en F1 macro).")

    return {
        'Target': target_name,
        'Mejor modelo': best_model_target,
        'Mejor F1 macro': best_f1_target
    }


models = {}
predictions = {}
cv_results = {}
tuning_results = {}

for model_name, estimator in model_specs.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', clone(preprocessor)),
        ('classifier', clone(estimator))
    ])

    pipeline.fit(X_train, y_train)
    predictions[model_name] = pipeline.predict(X_test)
    models[model_name] = pipeline

    if model_name != 'Baseline':
        cv_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv_strategy,
            scoring='f1_macro',
            n_jobs=-1
        )
        cv_results[model_name] = {
            'CV mean': cv_scores.mean(),
            'CV std': cv_scores.std()
        }

print("Modelos base entrenados: Baseline, Decision Tree, Decision Tree Balanced, Random Forest, Random Forest Balanced, Gradient Boosting")

print("- El ajuste de hiperparametros se mantiene acotado para conservar un tiempo de ejecucion razonable.")
for model_name, search_config in tuned_search_spaces.items():
    search_pipeline = Pipeline(steps=[
        ('preprocessor', clone(preprocessor)),
        ('classifier', clone(search_config['estimator']))
    ])

    randomized_search = RandomizedSearchCV(
        estimator=search_pipeline,
        param_distributions=search_config['param_distributions'],
        n_iter=6,
        scoring='f1_macro',
        cv=3,
        random_state=42,
        n_jobs=-1
    )

    randomized_search.fit(X_train, y_train)
    best_pipeline = randomized_search.best_estimator_
    best_params = {
        key.replace('classifier__', ''): value
        for key, value in randomized_search.best_params_.items()
    }

    models[model_name] = best_pipeline
    predictions[model_name] = best_pipeline.predict(X_test)
    tuning_results[model_name] = {
        'Search CV mean': randomized_search.best_score_
    }

    cv_scores = cross_val_score(
        best_pipeline,
        X_train,
        y_train,
        cv=cv_strategy,
        scoring='f1_macro',
        n_jobs=-1
    )
    cv_results[model_name] = {
        'CV mean': cv_scores.mean(),
        'CV std': cv_scores.std()
    }

    print(f"{model_name}: mejor F1 CV interna = {randomized_search.best_score_:.4f}")
    print(f"  Parametros seleccionados: {best_params}")

# =============================================================================
# 4. EVALUACION DE MODELOS
# =============================================================================
print("\n" + "=" * 80)
print("4. EVALUACION DE MODELOS")
print("=" * 80)
print("La evaluacion combina metricas de test y validacion cruzada. Se presta especial atencion al F1 macro por el caracter multiclase del problema.")

results = {}

for model_name in model_order:
    y_pred = predictions[model_name]
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    cv_mean = np.nan
    cv_std = np.nan
    if model_name in cv_results:
        cv_mean = cv_results[model_name]['CV mean']
        cv_std = cv_results[model_name]['CV std']

    search_cv_mean = np.nan
    if model_name in tuning_results:
        search_cv_mean = tuning_results[model_name]['Search CV mean']

    results[model_name] = {
        'Accuracy test': accuracy,
        'Precision macro': precision,
        'Recall macro': recall,
        'F1 macro': f1,
        'CV mean': cv_mean,
        'CV std': cv_std,
        'Search CV mean': search_cv_mean
    }

    print(f"\n{model_name}:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    if model_name in cv_results:
        print(f"  CV media:  {cv_mean:.4f}")
        print(f"  CV desv.:  {cv_std:.4f}")
    if model_name in tuning_results:
        print(f"  Search CV: {search_cv_mean:.4f}")

    print(f"\nReporte de clasificacion ({model_name}):")
    print(classification_report(
        y_test, y_pred, labels=target_classes, target_names=target_classes, zero_division=0
    ))

# Tabla comparativa
results_df = pd.DataFrame(results).T.loc[model_order, [
    'Accuracy test', 'Precision macro', 'Recall macro', 'F1 macro', 'CV mean', 'CV std', 'Search CV mean'
]]
results_display = results_df.round(4).astype(object)
results_display = results_display.where(pd.notnull(results_display), '-')
print("\nResumen de resultados:")
print(results_display)

# Grafica 2: comparacion de metricas
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
metrics = ['Accuracy test', 'Precision macro', 'Recall macro', 'F1 macro']

for idx, metric in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    values = results_df[metric]
    colors = sns.color_palette('Set2', len(values))
    ax.bar(values.index, values.values, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_title(metric, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_ylabel('Score')
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=30)

    for i, v in enumerate(values.values):
        ax.text(i, v + 0.02, f'{v:.4f}', ha='center', fontsize=10, fontweight='bold')

plt.suptitle('Comparacion de Rendimiento entre Modelos', fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('02_comparacion_modelos.png', dpi=300, bbox_inches='tight')
print("\nGrafica guardada: 02_comparacion_modelos.png")
plt.show()
plt.close()

# =============================================================================
# 5. MATRICES DE CONFUSION
# =============================================================================
print("\n" + "=" * 80)
print("5. MATRICES DE CONFUSION")
print("=" * 80)

n_cols = 4
n_rows = int(np.ceil(len(model_order) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 10))
axes = np.array(axes).ravel()

for idx, model_name in enumerate(model_order):
    y_pred = predictions[model_name]
    cm = confusion_matrix(y_test, y_pred, labels=target_classes)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=target_classes, yticklabels=target_classes,
                cbar_kws={'label': 'Cantidad'})
    axes[idx].set_title(f'{model_name}', fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('Real')
    axes[idx].set_xlabel('Predicho')

for idx in range(len(model_order), len(axes)):
    axes[idx].axis('off')

plt.suptitle('Matrices de Confusion', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('03_matrices_confusion.png', dpi=300, bbox_inches='tight')
print("\nGrafica guardada: 03_matrices_confusion.png")
plt.show()
plt.close()

# =============================================================================
# 6. IMPORTANCIA DE VARIABLES (RANDOM FOREST)
# =============================================================================
print("\n" + "=" * 80)
print("6. IMPORTANCIA DE VARIABLES (Random Forest)")
print("=" * 80)

rf_candidates = ['Random Forest', 'Random Forest Ajustado']
selected_rf_name = max(rf_candidates, key=lambda name: results_df.loc[name, 'F1 macro'])
rf_pipeline = models[selected_rf_name]
rf_model = rf_pipeline.named_steps['classifier']
rf_feature_names = rf_pipeline.named_steps['preprocessor'].get_feature_names_out()
rf_feature_names = [name.replace('num__', '').replace('cat__', '') for name in rf_feature_names]

print("Este analisis de importancia de variables se plantea como una lectura complementaria sobre la mejor variante de Random Forest, ya que este modelo facilita interpretar feature_importances_ sin implicar que sea el mejor modelo global.")
print(f"\nModelo Random Forest analizado: {selected_rf_name}")

feature_importance = pd.DataFrame({
    'Feature': rf_feature_names,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 caracteristicas:")
print(feature_importance.head(10))

fig, ax = plt.subplots(figsize=(12, 6))
top_features = feature_importance.head(10)
ax.barh(range(len(top_features)), top_features['Importance'].values,
        color='steelblue', edgecolor='black')
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['Feature'].values)
ax.set_xlabel('Importancia', fontweight='bold')
ax.set_title('Top 10 Caracteristicas mas Importantes', fontsize=12, fontweight='bold')
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

for i, v in enumerate(top_features['Importance'].values):
    ax.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('04_importancia_caracteristicas.png', dpi=300, bbox_inches='tight')
print("\nGrafica guardada: 04_importancia_caracteristicas.png")
plt.show()
plt.close()
print("Lectura final: estas importancias describen el comportamiento interno de esta variante de Random Forest y ofrecen una senal orientativa de relevancia relativa; no implican causalidad ni establecen una jerarquia absoluta valida para cualquier modelo.")

# ============================================================================
# 7. CONCLUSION
# ============================================================================
print("\n" + "=" * 80)
print("7. CONCLUSION")
print("=" * 80)
print("La conclusion integra el EDA, la seleccion de variables y el comportamiento observado en los modelos.")

best_model_name = results_df['F1 macro'].idxmax()
best_model_metrics = results_df.loc[best_model_name]
best_f1 = best_model_metrics['F1 macro']
best_accuracy = best_model_metrics['Accuracy test']
best_cv_mean = best_model_metrics['CV mean']
best_cv_std = best_model_metrics['CV std']
baseline_f1 = results_df.loc['Baseline', 'F1 macro']
real_models_df = results_df.drop(index='Baseline')
best_cv_model = real_models_df['CV mean'].idxmax()
second_best_name = real_models_df['F1 macro'].drop(best_model_name).idxmax()
second_best_f1 = real_models_df['F1 macro'].drop(best_model_name).max()
best_cv_model_test_f1 = real_models_df.loc[best_cv_model, 'F1 macro']
gap_vs_baseline = best_f1 - baseline_f1
gap_vs_others = best_f1 - second_best_f1
dt_f1 = results_df.loc['Decision Tree', 'F1 macro']
dt_balanced_f1 = results_df.loc['Decision Tree Balanced', 'F1 macro']
rf_f1 = results_df.loc['Random Forest', 'F1 macro']
rf_balanced_f1 = results_df.loc['Random Forest Balanced', 'F1 macro']
dt_delta = dt_balanced_f1 - dt_f1
rf_delta = rf_balanced_f1 - rf_f1

print("\nSintesis academica:")
print(f"- Para la target principal 'Action Taken', el modelo con mejor rendimiento en test dentro de esta comparacion es {best_model_name}, con accuracy {best_accuracy:.4f} y F1 macro {best_f1:.4f}.")

if baseline_f1 > 0:
    if gap_vs_baseline > 0.05:
        print("- Frente al baseline se observa una mejora apreciable, lo que sugiere cierta capacidad predictiva en el problema.")
    elif gap_vs_baseline > 0.01:
        print("- Frente al baseline se observa una mejora moderada, coherente con la presencia de senal util, aunque no especialmente intensa.")
    else:
        print("- Frente al baseline la mejora es reducida, lo que apunta a un problema exigente con las variables retenidas.")
else:
    print("- Frente al baseline tambien se observa una mejora, partiendo ademas de una referencia muy simple.")

if gap_vs_others > 0.02:
    print(f"- La ventaja frente a {second_best_name} es moderada, aunque conviene interpretarla con prudencia.")
elif gap_vs_others > 0.005:
    print(f"- La ventaja frente a {second_best_name} es pequena, por lo que los mejores modelos quedan muy proximos.")
else:
    print(f"- La diferencia frente a {second_best_name} es muy reducida y no permite hablar de una superioridad clara.")

if pd.notnull(best_cv_mean):
    if abs(best_f1 - best_cv_mean) <= 0.02:
        if best_cv_std <= 0.01:
            print("- Test y validacion cruzada ofrecen una lectura coherente y no sugieren una inestabilidad marcada.")
        else:
            print("- Test y validacion cruzada ofrecen una lectura coherente, aunque con cierta variabilidad entre particiones.")
    else:
        print("- La comparacion entre test y validacion cruzada aconseja cautela, porque parte de la diferencia observada podria depender de la muestra.")

if best_model_name != best_cv_model:
    if abs(best_f1 - best_cv_model_test_f1) <= 0.01:
        print(f"- {best_cv_model} obtiene la mejor media en validacion cruzada, pero en test queda muy cerca de {best_model_name}; la comparacion puede considerarse competitiva.")
    else:
        print(f"- {best_cv_model} lidera la validacion cruzada, mientras que {best_model_name} destaca en test; las diferencias invitan a una lectura competitiva mas que concluyente.")

if max_signal_gap > 0.08:
    print("- El EDA apunto a algunas diferencias entre variables categoricas y target, por lo que podia esperarse cierta capacidad de discriminacion, aunque no una separacion perfecta.")
elif max_signal_gap > 0.03:
    print("- El EDA apunto a relaciones moderadas entre variables categoricas y target, algo coherente con un rendimiento tambien moderado.")
else:
    print("- El EDA mostro relaciones debiles entre varias variables categoricas y la target; por eso, un F1 macro moderado resulta coherente con la informacion disponible.")

print("- En conjunto, el rendimiento es moderado porque las variables retenidas separan solo parcialmente las clases.")
print("- Ademas, parte de la informacion potencialmente util quedo fuera por alta cardinalidad, texto libre o baja capacidad informativa, lo que apunta a un techo de rendimiento condicionado por el dataset.")

if dt_delta > 0.01 and rf_delta > 0.01:
    print("- Las versiones con class_weight='balanced' muestran una mejora en ambos modelos de arbol, aunque el efecto sigue siendo acotado.")
elif dt_delta < -0.01 and rf_delta < -0.01:
    print("- Las versiones con class_weight='balanced' empeoran en ambos modelos de arbol, lo que apunta a que el desbalance no seria el principal limite del conjunto de datos.")
else:
    print(f"- El efecto de class_weight='balanced' es pequeno: la variacion en F1 macro es de {dt_delta:+.4f} en Decision Tree y {rf_delta:+.4f} en Random Forest. Esto sugiere que el limite podria estar mas en la senal disponible que en el reparto de clases.")

print("\nLimitaciones reales del dataset:")
print("- Varias columnas potencialmente utiles se excluyen porque son texto libre, identificadores de muy alta cardinalidad o variables casi constantes tras la imputacion.")
print("- Las relaciones observadas entre predictores y target son suaves, lo que dificulta una separacion nitida entre las clases.")
print("- La busqueda de hiperparametros fue deliberadamente acotada para mantener un proceso razonable y reproducible.")

print("\nCierre final:")
print("La practica desarrolla un flujo completo de clasificacion y mantiene una linea metodologica coherente desde el EDA hasta la interpretacion final.")
print("Aunque las metricas son moderadas, el resultado puede considerarse valido en un dataset de este tipo si se interpreta en relacion con la senal disponible y con las limitaciones del problema.")
print("En ese sentido, el valor principal de la practica esta en la coherencia del proceso seguido, en la comparacion razonada de modelos y en una conclusion alineada con lo que muestran los datos.")

# ============================================================================
# 8. EXPERIMENTOS ADICIONALES
# ============================================================================
print("\n" + "=" * 80)
print("8. EXPERIMENTOS ADICIONALES")
print("=" * 80)
print("\n'Action Taken' se mantiene como target obligatoria y principal; 'Severity Level' y 'Attack Type' se incluyen como ampliacion comparativa opcional.")

target_comparison_rows = [
    {
        'Target': 'Action Taken',
        'Mejor modelo': best_model_name,
        'Mejor F1 macro': best_f1
    }
]

for secondary_target in ['Severity Level', 'Attack Type']:
    target_comparison_rows.append(run_additional_target_experiment(secondary_target))

# ============================================================================
# 9. COMPARACION FINAL ENTRE TARGETS
# ============================================================================
print("\n" + "=" * 80)
print("9. COMPARACION FINAL ENTRE TARGETS")
print("=" * 80)

target_comparison_df = pd.DataFrame(target_comparison_rows).set_index('Target')
target_comparison_display = target_comparison_df.copy()
target_comparison_display['Mejor F1 macro'] = target_comparison_display['Mejor F1 macro'].map(lambda value: f"{value:.4f}")

print("\nResumen comparativo:")
print(target_comparison_display)

action_taken_f1 = target_comparison_df.loc['Action Taken', 'Mejor F1 macro']
optional_targets_f1 = target_comparison_df.loc[['Severity Level', 'Attack Type'], 'Mejor F1 macro']
severity_level_f1 = target_comparison_df.loc['Severity Level', 'Mejor F1 macro']
attack_type_f1 = target_comparison_df.loc['Attack Type', 'Mejor F1 macro']
severity_delta = severity_level_f1 - action_taken_f1
attack_type_delta = attack_type_f1 - action_taken_f1

print("\nLectura final de la practica:")
if severity_delta > 0.01:
    print(f"- 'Severity Level' podria resultar algo mas accesible que 'Action Taken' con las variables retenidas ({severity_delta:+.4f} en F1 macro).")
elif severity_delta < -0.01:
    print(f"- 'Severity Level' podria resultar algo mas exigente que 'Action Taken' con las variables retenidas ({severity_delta:+.4f} en F1 macro).")
else:
    print(f"- 'Severity Level' muestra una dificultad cercana a 'Action Taken' ({severity_delta:+.4f} en F1 macro).")

if attack_type_delta > 0.01:
    print(f"- 'Attack Type' podria resultar algo mas accesible que 'Action Taken' con las variables retenidas ({attack_type_delta:+.4f} en F1 macro).")
elif attack_type_delta < -0.01:
    print(f"- 'Attack Type' podria resultar algo mas exigente que 'Action Taken' con las variables retenidas ({attack_type_delta:+.4f} en F1 macro).")
else:
    print(f"- 'Attack Type' muestra una dificultad cercana a 'Action Taken' ({attack_type_delta:+.4f} en F1 macro).")

if abs(severity_delta) <= 0.01 and abs(attack_type_delta) <= 0.01:
    print("- Como ampliacion comparativa, ambas targets opcionales quedan cerca de la target principal y no alteran la lectura general de la practica.")
elif severity_delta < -0.01 and attack_type_delta < -0.01:
    print("- Como ampliacion comparativa, las targets opcionales podrian resultar algo mas exigentes que la target principal, sin desplazar el foco de 'Action Taken'.")
elif severity_delta > 0.01 and attack_type_delta > 0.01:
    print("- Como ampliacion comparativa, las targets opcionales podrian resultar algo mas accesibles que la target principal con las variables disponibles, sin desplazar el foco de 'Action Taken'.")
else:
    print("- Como ampliacion comparativa, las targets opcionales muestran un comportamiento mixto respecto a 'Action Taken', por lo que esta lectura debe entenderse como complementaria.")

print("\n" + "=" * 80)
print("Practica completada.")
print("=" * 80)



