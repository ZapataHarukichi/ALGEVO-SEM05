#1 traer el dataset
import pandas as pd
df = pd.read_csv("dataset5.csv", sep=";")

#2 -> heurística simple: asignación secuencial
# {'T01': ('S1','F1'), 'T02':('S1','F2')}
def heuristica_secuencial(df):
    asignacion = {}
    salas = ['S1','S2','S3','S4','S5','S6']
    franjas = df.columns[1:] #['F1','F2','F3','F4','F5','F6']

    #inicializar uso por sala y franja
    uso_sala_franja = {(sala, franja): 0 for sala in salas for franja in franjas}
    uso_por_sala = { sala: 0 for sala in salas}
    for idx, row in df.iterrows():
        tesista = row['TesistaID']
        asignado = False
        for franja in franjas:
            if row[franja] == 1:
                for sala in salas:
                    if uso_sala_franja[(sala, franja)] == 0 and uso_por_sala[sala] < 4:
                        asignacion[tesista] = (sala, franja)
                        uso_sala_franja[(sala, franja)] = 1
                        uso_por_sala[sala] += 1
                        asignado = True
                        break
                if asignado:
                    break
    return asignacion
        
asignacion = heuristica_secuencial(df)
print(asignacion)
#3 función de evaluación
def evaluar(asignacion):
    salas = ['S1','S2','S3','S4','S5','S6']
    franjas = ['F1','F2','F3','F4','F5','F6']

    # contadores
    solapamientos = 0
    uso_por_sala_franja = {}
    uso_por_sala = {sala: [] for sala in salas}

    for tesista, (sala, franja) in asignacion.items():
        clave = (sala, franja)
        uso_por_sala_franja[clave] = uso_por_sala_franja.get(clave, 0) + 1
        uso_por_sala[sala].append(franja)

    #contar solapamientos
    for (sala, franja), conteo in uso_por_sala_franja.items():
        if conteo > 1:
            solapamientos += conteo - 1
    # contar huecos y exceso de horas
    huecos = 0
    exceso_horas = 0
    for sala, franjas_usadas in uso_por_sala.items():
        indices = sorted([int(f[1]) for f in franjas_usadas])
        if len(indices) > 4:
            exceso_horas += len(indices) - 4
        if len(indices) >= 2:
            for i in range(indices[0], indices[-1] + 1):
                if i not in indices:
                    huecos += 1

    # Calcular penalización total
    costo_total = solapamientos * 100 + huecos * 10 + exceso_horas * 50
    return costo_total


import random

def generar_vecino(asignacion_actual, df):
    nuevo = asignacion_actual.copy()
    tesistas = list(nuevo.keys())
    tesista_a_mover = random.choice(tesistas)

    salas = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']
    franjas = df.columns[1:]  # ['F1', 'F2', ..., 'F6']

    # Obtener disponibilidad del tesista
    fila = df[df['TesistaID'] == tesista_a_mover].iloc[0]
    opciones_disponibles = [
        (sala, franja)
        for sala in salas
        for franja in franjas
        if fila[franja] == 1 and (sala, franja) != nuevo[tesista_a_mover]
    ]

    # Si hay al menos una nueva opción, hacer el cambio
    if opciones_disponibles:
        nueva_opcion = random.choice(opciones_disponibles)
        nuevo[tesista_a_mover] = nueva_opcion

    return nuevo
asignacion = heuristica_secuencial(df)
vecino = generar_vecino(asignacion, df)

print("Asignación original:", asignacion)
print("Vecino generado:", vecino)


def hill_climbing(df, max_iter=1000):
    actual = heuristica_secuencial(df)
    costo_actual = evaluar(actual)

    for i in range(max_iter):
        vecino = generar_vecino(actual, df)
        costo_vecino = evaluar(vecino)

        if costo_vecino < costo_actual:
            actual = vecino
            costo_actual = costo_vecino
            print(f"Iter {i}: Mejora encontrada con costo {costo_actual}")

    return actual, costo_actual

solucion_final, costo_final = hill_climbing(df)
print("\nAsignación final:")
for tesista, (sala, franja) in solucion_final.items():
    print(f"{tesista} → {sala}, {franja}")
print(f"\nCosto total final: {costo_final}")






