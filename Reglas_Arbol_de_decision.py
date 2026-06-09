'''

En este archivo extraemos las normas exactas que sigue el programa para cambiar un precio, escritas paso a paso para que cualquier trabajador pueda entenderlas.

'''


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

df = pd.read_csv('dataset_ml_listo.csv')

variables = ['stock_hour6_22_cnt', 'sale_amount', 'holiday_flag', 
             'avg_temperature', 'avg_humidity', 'avg_wind_level']
X = df[variables].fillna(0)
y = df['accion_precio']

# Entrenamos un árbol un poco más pequeño como ejemplo para poder leer las reglas perfectamente
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
modelo = DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=42)
modelo.fit(X_train, y_train)

# Traducimos el modelo a texto
reglas = export_text(modelo, feature_names=variables)
print(reglas)

print("\n Categorias:  Clase 0 (Mantener), Clase 1 (Descuento Leve), Clase 2 (Liquidación)")