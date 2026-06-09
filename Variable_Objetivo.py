'''

En este archivo convertimos los descuentos del supermercado en tres acciones: Mantener precio, Descuento Leve o Liquidación.

'''

import pandas as pd

print("Cargando el dataset")
df = pd.read_csv('dataset_tfg.csv')

# Creamos una función que define la regla de negocio del supermercado
def definir_accion_precio(descuento):
    if descuento == 1.0:
        return 0  # Mantener Precio
    elif descuento >= 0.85:
        return 1  # Descuento Leve
    else:
        return 2  # Liquidación

# Aplicamos la regla para crear nuestra columna objetivo
print("Generando la variable objetivo 'accion_precio'")
df['accion_precio'] = df['discount'].apply(definir_accion_precio)


print("\n Resumen de decisiones de precio:")
conteo = df['accion_precio'].value_counts()
print(f"0 - Mantener Precio: {conteo[0]} productos")
print(f"1 - Descuento Leve:  {conteo[1]} productos")
print(f"2 - Liquidación:     {conteo[2]} productos")

# Guardamos el dataset con la nueva variable objetivo que hemos creado
df.to_csv('dataset_ml_listo.csv', index=False)
