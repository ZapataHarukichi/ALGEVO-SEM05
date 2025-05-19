import numpy as np
import random

# Paso 1: Cargar matriz de distancias
# Puedes reemplazar esto con una lectura de CSV si tienes el archivo.
# Por ahora simulamos una matriz aleatoria válida:
# matriz[i][j] = distancia de i a j

np.random.seed(42)
matriz = np.random.randint(10, 100, size=(10, 10))
np.fill_diagonal(matriz, 0)  # Distancia de un nodo a sí mismo es 0

# Paso 2: Calcular distancia total de una ruta
def distancia_total(ruta, matriz):
    dist = 0
    for i in range(len(ruta)):
        dist += matriz[ruta[i]][ruta[(i + 1) % len(ruta)]]  # Cierra el ciclo
    return dist

# Paso 3: Generar vecino (intercambio de 2 nodos)
def generar_vecino(ruta):
    nueva_ruta = ruta[:]
    i, j = random.sample(range(len(ruta)), 2)
    nueva_ruta[i], nueva_ruta[j] = nueva_ruta[j], nueva_ruta[i]
    return nueva_ruta

# Paso 4: Hill Climbing
ruta_actual = list(range(10))
random.shuffle(ruta_actual)
mejor_distancia = distancia_total(ruta_actual, matriz)

for _ in range(1000):
    vecino = generar_vecino(ruta_actual)
    nueva_distancia = distancia_total(vecino, matriz)
    if nueva_distancia < mejor_distancia:
        ruta_actual = vecino
        mejor_distancia = nueva_distancia

# Paso 5: Resultados
print("Ruta óptima encontrada:", ruta_actual)
print("Distancia total:", mejor_distancia, "metros")
