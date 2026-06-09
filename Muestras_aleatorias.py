'''

En este archivo seleccionamos una parte del dataset para que el ordenador pueda analizarlos en un tiempo razonable.

'''


import pandas as pd
from datasets import load_dataset

archivo_salida = 'dataset_tfg.csv'

# Descargamos el dataset
dataset = load_dataset('Dingdong-Inc/FreshRetailNet-50K')
df = dataset['train'].to_pandas()

print(f"Dataset cargado: {len(df)} filas.")
print("Extrayendo una muestra aleatoria de 50.000 filas")

# Extraemos la muestra fijando la semilla (random_state=42) para que sea reproducible
df_muestra = df.sample(n=50000, random_state=42)


df_muestra.to_csv(archivo_salida, index=False)

print("El archivo ha sido creado")