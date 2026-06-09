'''

En este archivo buscamos qué relación existe entre variables del dataset.

'''

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('dataset_ml_listo.csv')

# Elegimos las variables que queremos comparar matemáticamente
variables = ['stock_hour6_22_cnt', 'sale_amount', 'discount', 
             'avg_temperature', 'avg_humidity', 'avg_wind_level', 'accion_precio']
df_numerico = df[variables]

print("Calculando la matriz de correlación")
matriz_correlacion = df_numerico.corr()

# Configuramos el tamaño
plt.figure(figsize=(10, 8))

# Dibujamos el mapa de calor,
# utilizamos annot=True para poner los números dentro y cmap='coolwarm' para poner los colores en este caso rojo y azul
sns.heatmap(matriz_correlacion, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)


plt.title('Mapa de Calor: Correlación de Variables del Supermercado', fontsize=14)
plt.tight_layout()

# Guardamos la imagen para adjuntarla en la memoria
nombre_imagen = 'mapa_correlaciones.png'
plt.savefig(nombre_imagen, dpi=300)

# Mostramos la gráfica en pantalla
plt.show()