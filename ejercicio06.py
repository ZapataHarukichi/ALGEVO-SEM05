import pandas as pd
df = pd.read_csv("dataset6.csv", sep=";")

print(df.head())

#Solución inicial
#2 bitstring de paso 2
def evaluar(bitstring, df, time_max =90, difficulty_min=180, difficulty_max=200):
    time_total = 0
    difficulty_total = 0
    #acumular los tiempos y las difcultades

    for i, bit in enumerate(bitstring):
        if bit == '1':
            time_total += df.loc[i,'Time_min']
            difficulty_total += df.loc[i,'Difficulty']


    if (time_total > time_max or difficulty_total < difficulty_min or difficulty_total > difficulty_max):
        return -float('inf') #penalización fuerte
    else:
        return difficulty_total
    

#bitstring = '10101010'
#aptitud = evaluar(bitstring, df)
#print(f"Aptitud de {bitstring}: {aptitud}")


#bitstring = '10101'
#aptitud_time, aptitud_difficulty = evaluar(bitstring, df, time_max =90, difficulty_max=200)
#print(aptitud_time, " ", aptitud_difficulty)

# 3 -> Generar vecino 
import random
def generar_vecino(bitstring):
    lista_bits = list(bitstring)
    idx = random.randint(0, len(lista_bits) - 1)

    #voltear el bit
    #lista_bits[idx] = '1' if lista_bits[idx] == '1' else '0'
    lista_bits[idx] = '1' if lista_bits[idx] == '0' else '0'
    #volver a unir en string

    vecino = ''.join(lista_bits)
    return vecino


actual = '10101'
vecino = generar_vecino(actual)
print(f"Solución actual: {actual}")
print(f"Vecino generado: {vecino}")


# generar una solución inicial válida
def construir_solucion_valida(df, time_max =90, difficulty_min=180, difficulty_max=200):
    n = len(df)
    indices = list(range(n))
    random.shuffle(indices) #?

    bitstring = ['0'] * n
    total_time = 0
    total_difficulty = 0
    
    for i in indices:
        tiempo = df.loc[i, 'Time_min']
        dificultad = df.loc[i, 'Difficulty']

        if total_time + tiempo <= time_max and total_difficulty + dificultad <= difficulty_max: ##cambie difficulty
            bitstring[i] = '1'
            total_time += tiempo
            total_difficulty += dificultad

            if total_difficulty >= difficulty_min:
                break 
    return ''.join(bitstring)      


def hill_climbing(df, time_max =90, difficulty_min=180, difficulty_max=200, max_iter = 1000):
    # Solución inicial aleatoria
    actual_bitstring = construir_solucion_valida(df, time_max =90, difficulty_min=180, difficulty_max=200)
    aptitud_actual = evaluar(actual_bitstring, df, time_max =90,difficulty_min=180 ,difficulty_max=200)
    #actual=''.join(random.choice(['0','1']) for _ in range(n))
    #actual_aptitud = evaluar(actual, df, presupuesto)

    for i in range(max_iter):
        vecino = generar_vecino(actual_bitstring)
        aptitud_vecino = evaluar(vecino, df, time_max =90,difficulty_min=180 ,difficulty_max=200)
        if aptitud_vecino > aptitud_actual:
            actual_bitstring = vecino
            aptitud_actual = aptitud_vecino
            print(f"Iter {i}: Mejor aptitud: {aptitud_actual}")
    return actual_bitstring, aptitud_actual

#5 -> mostrar las preguntas seleccionadas a partir de un bitstring
def preguntas_seleccionadas(bitstring):
    seleccionados = []
    tiempo_total = 0
    dificultad_total = 0
    for i, bit in enumerate(bitstring):
        if bit == '1':
            seleccionados.append(df.loc[i, 'QuestionID'])
            tiempo_total += df.loc[i, 'Time_min']
            dificultad_total += df.loc[i, 'Difficulty']
    return seleccionados, tiempo_total, dificultad_total

# Ejecutar
solucion, dificultad = hill_climbing(df)
seleccionadas, tiempo_total, dificultad_total = preguntas_seleccionadas(solucion)
#tengo un problema que el tiempo y la dificultad superan siempre mi tiempo máximo y dificultad, creo que tendrían que elegirse pocas preguntas
print("Preguntas seleccionadas: ", seleccionadas)
print("Tiempo total del examen: ", tiempo_total, " min")
print("Dificultad total del examen: ", dificultad_total)


