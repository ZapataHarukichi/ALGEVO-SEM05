import pandas as pd
import random

# Paso 1: Cargar la tabla
df = pd.read_csv("dataset2.csv", sep=";")
mentores = df.index.tolist()
horarios = df.columns.tolist()
num_mentores = len(df)

# Paso 2: Generar una solución inicial válida
def generar_solucion_inicial():
    solucion = []
    for i in range(num_mentores):
        while True:
            inicio = random.randint(0, len(horarios) - 2)
            if df.iloc[i, inicio] == 1 and df.iloc[i, inicio + 1] == 1:
                solucion.append(inicio)
                break
    return solucion

# Paso 3: Función de costo
def calcular_choques(solucion):
    choques = 0
    for i, inicio in enumerate(solucion):
        if df.iloc[i, inicio] == 0:
            choques += 1
        if df.iloc[i, inicio + 1] == 0:
            choques += 1
    return choques

# Paso 4: Vecindario (cambiar horario de un mentor)
def generar_vecino(solucion):
    nuevo = solucion[:]
    i = random.randint(0, num_mentores - 1)
    intentos = 0
    while intentos < 10:
        nuevo_inicio = random.randint(0, len(horarios) - 2)
        if df.iloc[i, nuevo_inicio] == 1 and df.iloc[i, nuevo_inicio + 1] == 1:
            nuevo[i] = nuevo_inicio
            break
        intentos += 1
    return nuevo

# Paso 5: Búsqueda local
solucion = generar_solucion_inicial()
costo = calcular_choques(solucion)

while costo > 0:
    vecino = generar_vecino(solucion)
    nuevo_costo = calcular_choques(vecino)
    if nuevo_costo < costo:
        solucion = vecino
        costo = nuevo_costo

# Paso 6: Mostrar resultados
print("Asignación final de bloques (índice de hora inicial):", solucion)
print("Choques:", costo)
