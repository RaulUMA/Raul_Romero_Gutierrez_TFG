'''

En este archivo investigamos en qué situaciones concretas falla la Inteligencia Artificial.

'''


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('dataset_ml_listo.csv')

variables = ['stock_hour6_22_cnt', 'sale_amount', 'holiday_flag', 
             'avg_temperature', 'avg_humidity', 'avg_wind_level']
X = df[variables].fillna(0)
y = df['accion_precio']

# Entrenamos el modelo Árbol de Decisión
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
modelo = DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=42)
modelo.fit(X_train, y_train)
predicciones = modelo.predict(X_test)

# Creamos un DataFrame para analizar los resultados
df_resultados = X_test.copy()
df_resultados['Real'] = y_test
df_resultados['Prediccion'] = predicciones

# Marcamos con True donde el modelo se ha equivocado
df_resultados['Error'] = df_resultados['Real'] != df_resultados['Prediccion']

# 1. Gráfica: Errores por tipo de producto
plt.figure(figsize=(8, 5))
sns.boxplot(x='Error', y='sale_amount', data=df_resultados, palette='Set2')
plt.title('Distribución de Volumen de Ventas en Aciertos vs Errores', fontsize=12)
plt.xticks([0, 1], ['Acierto (False)', 'Error (True)'])
plt.ylabel('Volumen de Ventas Matinales')
plt.tight_layout()
plt.savefig('error_rotacion.png', dpi=300)
plt.close()

# 2. Gráfica: Errores por clima
plt.figure(figsize=(8, 5))
sns.histplot(data=df_resultados, x='avg_temperature', hue='Error', multiple="stack", palette='Set1', bins=20)
plt.title('Distribución de Errores según la Temperatura Exterior', fontsize=12)
plt.xlabel('Temperatura Media (°C)')
plt.ylabel('Cantidad de Predicciones')
plt.tight_layout()
plt.savefig('error_temperatura.png', dpi=300)
plt.close()

# 3. Ejemplos concretos de predicciones fallidas
fallos = df_resultados[df_resultados['Error'] == True].head(3)
print(fallos[['stock_hour6_22_cnt', 'sale_amount', 'avg_temperature', 'Real', 'Prediccion']])