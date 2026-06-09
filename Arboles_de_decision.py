'''

En este archivo enseñamos a la Inteligencia Artificial a recomendar precios utilizando el algoritmo "Árboles de Decisión".

'''



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('dataset_ml_listo.csv')

# Elegimos las variables que el modelo usará para estudiar, que es la variable "X" y la variable "Y" que es la que queremos predecir
variables_entrenamiento = ['stock_hour6_22_cnt', 'sale_amount', 'holiday_flag', 
                           'avg_temperature', 'avg_humidity', 'avg_wind_level']
X = df[variables_entrenamiento].fillna(0)
y = df['accion_precio']

# Dividimos los datos: 80% para entrenar, 20% para el examen final
print("Dividiendo los datos en Entrenamiento (80%) y Test (20%)")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creamos el Árbol de Decisión,
# ponemos una profundidad máxima de 10 (max_depth=10) para no crear un arból muy grande y sea imposible de comprender.
modelo_arbol = DecisionTreeClassifier(max_depth=10,class_weight='balanced', random_state=42)


print("Entrenando al Árbol de Decisión")
modelo_arbol.fit(X_train, y_train)

# Ahora estudiamos los resultados con Xtest
print("Haciendo predicciones en los datos de test")
predicciones = modelo_arbol.predict(X_test)

# Comprobamos la precisión del Árbol de Decisión
precision = accuracy_score(y_test, predicciones)
print(f" Precisión (Accuracy) del Árbol de Decisión: {precision * 100:.2f}%")

# Vemos los aciertos de cada clase
print("Aciertos divididos por categoria:")
print(classification_report(y_test, predicciones, target_names=['Mantener (0)', 'Desc. Leve (1)', 'Liquidación (2)'], zero_division = 0))