'''

En este archivo generamos un gráfico que nos dice en qué se fija más la Inteligencia Artificial para cambiar un precio.

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

# Entrenamos el modelo ganador, en este caso el Arbol de Decisión
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
modelo = DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=42)
modelo.fit(X_train, y_train)

# Extraemos la importancia de cada variable
importancias = pd.DataFrame({'Variable': variables, 'Importancia': modelo.feature_importances_})
importancias = importancias.sort_values(by='Importancia', ascending=False)

# Dibujamos la gráfica
plt.figure(figsize=(10, 6))
sns.barplot(x='Importancia', y='Variable', data=importancias, palette='magma')
plt.title('Importancia de las Variables en la Decisión de Precios', fontsize=14)
plt.xlabel('Peso en el modelo (0 a 1)', fontsize=12)
plt.ylabel('Variable Crítica', fontsize=12)
plt.tight_layout()

# Guardamos la imagen
plt.savefig('importancia_variables.png', dpi=300)

# Lo mostramos por pantalla
plt.show()