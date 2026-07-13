# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando componentes principales.
#   El pca usa todas las componentes.
# - Escala la matriz de entrada al intervalo [0, 1].
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una red neuronal tipo MLP.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#
import os
import pandas as pd
import gzip
import pickle
import json
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)



def limpiar_datos_credito(ruta_entrada, ruta_salida):
    
    df = pd.read_csv(ruta_entrada, index_col=False, compression="zip")

    if "default payment next month" in df.columns:
        df = df.rename(columns={"default payment next month": "default"})

    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    # Eliminamos únicamente los ceros de educación y matrimonio
    if "EDUCATION" in df.columns:
        df = df[df["EDUCATION"] != 0]
    if "MARRIAGE" in df.columns:
        df = df[df["MARRIAGE"] != 0]

    # Agrupamos los valores mayores a 4 (5 y 6) en la categoría 4
    if "EDUCATION" in df.columns:
        df.loc[df["EDUCATION"] > 4, "EDUCATION"] = 4
        
    df = df.dropna()

    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    df.to_csv(ruta_salida, index=False)

    return df

def datasets_x_y(dataset1,dataset2,columna_objetivo):
    df_train = pd.read_csv(dataset1)
    df_test = pd.read_csv(dataset2)

    X_train = df_train.drop(columns=[columna_objetivo])
    y_train = df_train[columna_objetivo]

    X_test = df_test.drop(columns=[columna_objetivo])
    y_test = df_test[columna_objetivo]

    return [X_train, y_train, X_test, y_test]


def crear_y_optimizar_pipeline(X_train, y_train):
    categorical = ["SEX", "EDUCATION", "MARRIAGE"]
    numerical = [num for num in X_train.columns if num not in categorical]

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(), categorical),
        ("num", StandardScaler(), numerical),
    ], remainder='drop')

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("feature_selection", SelectKBest(score_func=f_classif)),
        ("pca", PCA()),
        ("classifier", MLPClassifier(
            max_iter=15000, 
            random_state=21, 
            #early_stopping=False  # Permite converger por completo
        )),
    ])

    param_grid = {
        'pca__n_components': [20],
        'feature_selection__k': [20],
        'classifier__hidden_layer_sizes': [(50, 30, 40, 60)],
        'classifier__alpha': [0.26],          
        'classifier__learning_rate_init': [0.001],
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=-1,
        refit=True,
    )

    grid_search.fit(X_train, y_train)

    return grid_search


def guardar_modelo_comprimido(grid_search, ruta_salida):
    
    directorio = os.path.dirname(ruta_salida)
    if directorio:
        os.makedirs(directorio, exist_ok=True)

    with gzip.open(ruta_salida, "wb") as archivo:
        pickle.dump(grid_search, archivo)

def calcular_y_guardar_metricas(mejor_modelo, X_train, y_train, X_test, y_test, ruta_salida):
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)

    y_train_pred = mejor_modelo.predict(X_train)
    y_test_pred = mejor_modelo.predict(X_test)

    lineas_json = []

    # metricas para train
    metricas_train = {
        "type": "metrics",
        "dataset": "train",
        "precision": float(precision_score(y_train, y_train_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_train, y_train_pred)),
        "recall": float(recall_score(y_train, y_train_pred, zero_division=0)),
        "f1_score": float(f1_score(y_train, y_train_pred, zero_division=0)),
    }
    lineas_json.append(metricas_train)

    # metricas para test
    metricas_test = {
        "type": "metrics",
        "dataset": "test",
        "precision": float(precision_score(y_test, y_test_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_test_pred)),
        "recall": float(recall_score(y_test, y_test_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_test_pred, zero_division=0)),
    }
    lineas_json.append(metricas_test)

    # matrices de confusiónq
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_train_dict = {
        "type": "cm_matrix",
        "dataset": "train",
        "true_0": {
            "predicted_0": int(cm_train[0, 0]),
            "predicted_1": int(cm_train[0, 1]),
        },
        "true_1": {
            "predicted_0": int(cm_train[1, 0]),
            "predicted_1": int(cm_train[1, 1]),
        },
    }
    lineas_json.append(cm_train_dict)

    cm_test = confusion_matrix(y_test, y_test_pred)
    cm_test_dict = {
        "type": "cm_matrix",
        "dataset": "test",
        "true_0": {
            "predicted_0": int(cm_test[0, 0]),
            "predicted_1": int(cm_test[0, 1]),
        },
        "true_1": {
            "predicted_0": int(cm_test[1, 0]),
            "predicted_1": int(cm_test[1, 1]),
        },
    }
    lineas_json.append(cm_test_dict)

    with open(ruta_salida, "w", encoding="utf-8") as f:
        for linea in lineas_json:
            f.write(json.dumps(linea) + "\n")


if __name__ == "__main__":
    # paso 1: limpieza de db's
    ruta_train_zip = "files/input/train_data.csv.zip"
    ruta_test_zip = "files/input/test_data.csv.zip"

    ruta_train_salida = "files/output/train_cleaned.csv"
    ruta_test_salida = "files/output/test_cleaned.csv"

    df_train_limpio = limpiar_datos_credito(ruta_train_zip, ruta_train_salida)
    df_test_limpio = limpiar_datos_credito(ruta_test_zip, ruta_test_salida)

    # paso 2: division de datasets
    X_train, y_train, X_test, y_test = datasets_x_y(
        ruta_train_salida, ruta_test_salida, "default"
    )

    # paso 3 y 4: creacion y optimizacion de pipeline
    grid_searched = crear_y_optimizar_pipeline(X_train, y_train)

    # paso 5: guardar modelo optimizado en formato gzip
    guardar_modelo_comprimido(grid_searched, "files/models/model.pkl.gz")

    # paso 6 y 7: creacion y guardado de metricas de rendimiento
    calcular_y_guardar_metricas(
        grid_searched.best_estimator_,
        X_train,
        y_train,
        X_test,
        y_test,
        "files/output/metrics.json",
    )

    