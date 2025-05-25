from deap import base, creator, tools
import random

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness = creator.FitnessMax)

toolbox = base.Toolbox()

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

#1<- leer el CSV
df = pd.read_csv("dataset9.csv", sep=";")
X = df[['Feature1', 'Feature2','Feature3','Feature4','Feature5']].values
y_true= df['Spam'].values

# división de datos
X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.2, random_state=42)

def evaluar_individuo(individuo, X, y):
    pesos = np.array(individuo[:5])
    umbral = individuo[5]
    scores = np.dot(X, pesos)
    y_pred = (scores >= umbral).astype(int)
    f1 = f1_score(y, y_pred)
    return (f1,)

#3 <- hill climbing por individuo tras la mutación
import copy

def hill_climbing_individual(individuo, X, y, pasos=10, pasos_mutacion=0.05):
    mejor = copy.deepcopy(individuo)
    mejor_score = evaluar_individuo(mejor, X, y) [0]

    for _ in range(pasos):
        vecino = copy.deepcopy(mejor)
        idx = random.randint(0, 5)
        cambio = random.uniform(-pasos_mutacion, pasos_mutacion)
        vecino[idx] += cambio

        vecino[:5] = np.clip(vecino[:5], 0, 1)
        vecino[5] = np.clip(vecino[5], 0, 5)

        score = evaluar_individuo(vecino, X, y)[0]
        if score > mejor_score:
            mejor = vecino
            mejor_score = score

    return mejor

toolbox.register("attr_float", lambda: random.uniform(0, 1))
toolbox.register("attr_threshold", lambda: random.uniform(0, 5))
toolbox.register("individual", tools.initCycle, creator.Individual,
                (toolbox.attr_float,)*5 + (toolbox.attr_threshold,), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", lambda ind: evaluar_individuo(ind, X_train, y_train))
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.6)
toolbox.register("select", tools.selTournament, tournsize=3)

def evolucion(ngen=30, pop_size=20):
    pop = toolbox.population(n=pop_size)
    f1_scores = []
    for gen in range(ngen):
        offspring = []
        for ind in pop:
            ind_ = toolbox.clone(ind)
            toolbox.mutate(ind_)
            ind_ = hill_climbing_individual(ind_, X_train, y_train)
            ind_.fitness.values = evaluar_individuo(ind_, X_train, y_train)
            offspring.append(ind_)
        
        pop = toolbox.select(offspring, k=pop_size)
        fits = [ind.fitness.values[0] for ind in pop]
        f1_scores.append(max(fits))
        print(f"Generación {gen}: mejor F1 = {max(fits):.4f}")

    best = tools.selBest(pop, k=1)[0]
    f1_test = evaluar_individuo(best, X_test, y_test) [0]
    print(f"Mejor individuo en test: F1 = {f1_test:.4f}")
    return best, f1_scores 

mejor_ind, f1_por_gen = evolucion()
pesos = mejor_ind[:5]
umbral = mejor_ind[5]
print("Pesos óptimos:", pesos)
print("Umbral óptimo:", umbral)
print("F1 final:", evaluar_individuo(mejor_ind, X_train, y_train) [0])


