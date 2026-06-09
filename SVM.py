'''

En este archivo enseñamos a la Inteligencia Artificial a recomendar precios utilizando el algoritmo "Support Vector Machine"

'''


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv('dataset_ml_listo.csv')

# Elegimos las variables que el modelo usará para estudiar, que es la variable "X" y la variable "Y" que es la que queremos predecir
variables_entrenamiento = ['stock_hour6_22_cnt', 'sale_amount', 'holiday_flag', 
                           'avg_temperature', 'avg_humidity', 'avg_wind_level']
X = df[variables_entrenamiento].fillna(0)
y = df['accion_precio']

# Dividimos los datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalamos los datos para entrenar el algoritmo
print(" Escalando los datos")
scaler = StandardScaler()
X_train_escalado = scaler.fit_transform(X_train)
X_test_escalado = scaler.transform(X_test)

# Creamos el modelo SVM,
# usamos class_weight='balanced' para que tome en cuenta el descuento leve y no siempre mantenga el precio
modelo_svm = SVC(kernel='rbf', class_weight='balanced', random_state=42)


# Como SVM tarda bastante más, realizamos el escalado con 10000 datos solo
modelo_svm.fit(X_train_escalado[:10000], y_train[:10000])

print("Hacemos las predicciones ahora en test")
predicciones_svm = modelo_svm.predict(X_test_escalado)

precision_svm = accuracy_score(y_test, predicciones_svm)
print(f" Precisión (Accuracy) del SVM: {precision_svm * 100:.2f}%")

print("\n Cantidad de aciertos por categoria:")
print(classification_report(y_test, predicciones_svm, 
                            target_names=['Mantener (0)', 'Desc. Leve (1)', 'Liquidación (2)'], 
                            zero_division=0))