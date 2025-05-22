import pandas as pd
import random
df = pd.read_csv("dataset7.csv", sep=";")
#print(df.head())

#1->representar solución como lista de índices
def generar_solucion_inicial(df, num_equipos=5, tamaño_equipo=4):
    indices = list(range(len(df))) # qué hace range y qué hace len
    random.shuffle(indices)

    solucion = []
    for i in range(num_equipos):
        equipo = indices[i*tamaño_equipo : (i+1) * tamaño_equipo] # qué hace este código?
        solucion.append(equipo)
    return solucion

#solucion = generar_solucion_inicial(df)
#print(solucion)

#2-> función de aptitud
import numpy as np
from collections import Counter
def calcular_aptitud(df, solucion, penalizacion_habilidad=0.5):
    gpas_por_equipo = []
    penalizacion_total = 0
    
    for equipo in solucion:
        gpas = []
        habilidades = []

        for idx in equipo:
            gpas.append(df.loc[idx, 'GPA'])
            habilidades.append(df.loc[idx, 'Skill'])

        # varianza del GPA dentro del equipo
        varianza = np.var(gpas)
        gpas_por_equipo.append(varianza)

        #penalización si hay habilidades repetidas
        conteo = Counter(habilidades)
        repeticiones = sum(v - 1 for v in conteo.values() if v > 1)
        penalizacion_total += penalizacion_habilidad * repeticiones

    suma_varianzas = sum(gpas_por_equipo)
    aptitud_total = suma_varianzas + penalizacion_total

    return aptitud_total

solucion = generar_solucion_inicial(df)
aptitud = calcular_aptitud(df, solucion)
#print("Aptitud de la solución", aptitud)

import copy
#3-> generar vecino: swap de dos alumnos de equipo distinto
def generar_vecino(df, solucion, num_equipos = 5):
    #copiamos la solución para no modificar la original
    vecino = copy.deepcopy(solucion)

    #elegimos dos equipos distintos al azar
    equipo1, equipo2 = random.sample(range(num_equipos), 2)

    #elegimos un alumno al azar de cada equipo
    alumno1_idx = random.randint(0, len(vecino[equipo1]) - 1)
    alumno2_idx = random.randint(0, len(vecino[equipo2]) - 1)

    #intercambiamos los alumnos
    vecino[equipo1][alumno1_idx], vecino[equipo2][alumno2_idx] = (
        vecino[equipo2][alumno2_idx],
        vecino[equipo1][alumno1_idx]
    )
    return vecino

#solucion = generar_solucion_inicial(df)
#vecino = generar_vecino(df, solucion)
#print("Solución original:", solucion)
#print("Vecino generado:", vecino)

#4-> hill climbing
def hill_climbing(df, num_equipos=5, tamaño_equipo=4, max_iter=1000):
    actual = generar_solucion_inicial(df, num_equipos, tamaño_equipo)
    aptitud_actual = calcular_aptitud(df,actual)

    for _ in range(max_iter):
        vecino = generar_vecino(df, actual, num_equipos)
        aptitud_vecino = calcular_aptitud(df, vecino)

        if aptitud_vecino < aptitud_actual:
            actual = vecino
            aptitud_actual = aptitud_vecino

    return actual, aptitud_actual

def mostrar_metricas_equipo(df, solucion):
    for i, equipo in enumerate(solucion):
        gpas = [df.loc[idx, 'GPA'] for idx in equipo]
        habilidades = [df.loc[idx, 'Skill'] for idx in equipo]
        varianza_gpa = np.var(gpas)
        promedio_gpa = np.mean(gpas)
        conteo_skills = Counter(habilidades)

        print(f"Equipo {i+1}:")
        print(f"Miembros: {[df.loc[idx, 'StudentID'] for idx in equipo]}")
        print(f"Promedio GPA: {promedio_gpa:.2f}")
        print(f"Varianza GPA: {varianza_gpa:.4f}")
        print(f"Habilidades: {dict(conteo_skills)}")

#5-> mostrar resultado
mejor_solucion, mejor_aptitud = hill_climbing(df)
print("Mejor aptitud encontrada:", mejor_aptitud)
mostrar_metricas_equipo(df, mejor_solucion)
#for i, equipo in enumerate(mejor_solucion):
    #print(f"Equipo {i+1}: {[df.loc[idx, 'StudentID'] for idx in equipo]}")

