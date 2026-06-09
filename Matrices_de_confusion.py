'''

En este archivo creamos los gráficos que nos permiten ver de forma clara cuántas veces acierta cada modelo y en qué tipo de situaciones se equivoca más.

'''


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

df = pd.read_csv('dataset_ml_listo.csv')

# seleccionamos las variables que necesitamos
variables = ['stock_hour6_22_cnt', 'sale_amount', 'holiday_flag', 
             'avg_temperature', 'avg_humidity', 'avg_wind_level']
X = df[variables].fillna(0)
y = df['accion_precio']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
nombres_clases = ['Mantener', 'Desc. Leve', 'Liquidación']

# ARBOL DE DECISIÓN

print("Entrenando Árbol y dibujando su matriz")

modelo_arbol = DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=42)
modelo_arbol.fit(X_train, y_train)
predicciones_arbol = modelo_arbol.predict(X_test)

matriz_arbol = confusion_matrix(y_test, predicciones_arbol)

plt.figure(figsize=(6, 5))


sns.heatmap(matriz_arbol, annot=True, fmt='d', cmap='Blues', 
            xticklabels=nombres_clases, yticklabels=nombres_clases)
plt.title('Matriz de Confusión: Árbol de Decisión', fontsize=12)
plt.ylabel('Valor Real', fontweight='bold')
plt.xlabel('Predicción', fontweight='bold')
plt.tight_layout()
plt.savefig('matriz_arbol.png', dpi=300)
plt.close()

# SVM

print("Entrenando SVM y dibujando su matriz")

# Como SVM tarda bastante más, realizamos el escalado con 10000 datos solo
scaler = StandardScaler()
X_train_esc = scaler.fit_transform(X_train[:10000])
X_test_esc = scaler.transform(X_test)

modelo_svm = SVC(kernel='rbf', class_weight='balanced', random_state=42)
modelo_svm.fit(X_train_esc, y_train[:10000])
predicciones_svm = modelo_svm.predict(X_test_esc)

matriz_svm = confusion_matrix(y_test, predicciones_svm)

plt.figure(figsize=(6, 5))
sns.heatmap(matriz_svm, annot=True, fmt='d', cmap='Oranges', 
            xticklabels=nombres_clases, yticklabels=nombres_clases)

plt.title('Matriz de Confusión: SVM', fontsize=12)
plt.ylabel('Valor Real', fontweight='bold')
plt.xlabel('Predicción', fontweight='bold')
plt.tight_layout()
plt.savefig('matriz_svm.png', dpi=300)
plt.close()

print("Se han generado las gráficas con exito")