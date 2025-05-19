import pandas as pd

# Leer el archivo CSV
df = pd.read_csv('dataset1.csv', sep=';')

# Calcular la nota final promedio de los tres parciales
df['Final'] = df[['Parcial1', 'Parcial2', 'Parcial3']].mean(axis=1)


def aptitud(offset, notas_originales):
    notas_ajustadas = notas_originales + offset
    promedio = notas_ajustadas.mean()
    aprobados = (notas_ajustadas >= 11).sum()
    porcentaje_aprobados = aprobados / len(notas_ajustadas)

    if promedio > 14:
        return porcentaje_aprobados - (promedio - 14)  # Penalización
    else:
        return porcentaje_aprobados


import numpy as np
import random

# Datos base
notas_originales = df['Final'].copy()

# Inicialización
mejor_offset = 0
mejor_aptitud = aptitud(mejor_offset, notas_originales)

# Hill climbing
for _ in range(1000):  # iteraciones
    nuevo_offset = mejor_offset + random.choice([-0.5, 0.5])
    if -5 <= nuevo_offset <= 5:
        nueva_aptitud = aptitud(nuevo_offset, notas_originales)
        if nueva_aptitud > mejor_aptitud:
            mejor_offset = nuevo_offset
            mejor_aptitud = nueva_aptitud


print(f"Offset óptimo encontrado: {mejor_offset:.2f}")
notas_ajustadas = notas_originales + mejor_offset
nuevo_promedio = notas_ajustadas.mean()
porcentaje_aprobados = (notas_ajustadas >= 11).mean() * 100

print(f"Nuevo promedio: {nuevo_promedio:.2f}")
print(f"Porcentaje de aprobados: {porcentaje_aprobados:.2f}%")
