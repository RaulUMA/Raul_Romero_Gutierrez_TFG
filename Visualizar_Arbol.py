'''

En este archivo creamos un dibujo sencillo que muestra visualmente el camino que sigue el ordenador para decidir si un producto debe ser rebajado o no.

'''


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

df = pd.read_csv('dataset_ml_listo.csv')

variables = ['stock_hour6_22_cnt', 'sale_amount', 'holiday_flag', 
             'avg_temperature', 'avg_humidity', 'avg_wind_level']
X = df[variables].fillna(0)
y = df['accion_precio']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reducimos la profundidad a 2 para que genere un máximo de 7 nodos.
# Esto hace que genere un ejemplo mucho mas sencillo para su explicación en la memoria
modelo_visual = DecisionTreeClassifier(max_depth=2, class_weight='balanced', random_state=42)
modelo_visual.fit(X_train, y_train)

plt.figure(figsize=(12, 8))
plot_tree(modelo_visual, 
          feature_names=variables, 
          class_names=['Mantener', 'Desc. Leve', 'Liquidación'], 
          filled=True, 
          rounded=True, 
          fontsize=12)

plt.title('Estructura lógica simplificada del Árbol de Decisión (Profundidad = 2)', fontsize=16, pad=20)
plt.tight_layout()

# Guardamos la imagen
plt.savefig('arbol_visual.png', dpi=300)
