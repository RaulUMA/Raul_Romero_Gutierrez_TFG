'''
En este archivo realizaremos un gráfico de barras en el que se mostrarán el equilibrio entre las tres categorias (Mantener, descuento leve y liquidación)

'''

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('dataset_ml_listo.csv')


plt.figure(figsize=(8, 6))

# Contamos cuántos productos hay de cada clase
conteo_clases = df['accion_precio'].value_counts().sort_index()

# Le ponemos los nombres de las categorias en lugar de 0, 1 y 2
nombres_clases = ['0 - Mantener', '1 - Desc. Leve', '2 - Liquidación']

# Dibujamos el gráfico de barras
sns.barplot(x=nombres_clases, y=conteo_clases.values, palette='viridis')


plt.title('Distribución de las Decisiones de Precio', fontsize=14)
plt.xlabel('Categoría de Acción', fontsize=12)
plt.ylabel('Número de Productos', fontsize=12)


for i, valor in enumerate(conteo_clases.values):
    plt.text(i, valor + 300, str(valor), ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()

# Guardamos la imagen
nombre_imagen = 'barras_categorias.png'
plt.savefig(nombre_imagen, dpi=300)

print(f"Gráfico generado")

# Lo mostramos por pantalla
plt.show()