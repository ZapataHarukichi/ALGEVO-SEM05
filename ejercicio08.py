#Objetivo: minimizar el RMSE
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# DEAP
from deap import base, creator, tools

# leer el CSV
df = pd.read_csv("dataset8.csv", sep=";")
X = df[['Rooms', 'Area_m2']].values
y = df['Price_Soles'].values

# división de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Minimizar el RSME -> por eso weights=(-1.0,)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
# limites de alpha en escala log (ej: log(alpha) entre -4 y 3 -> alpha entre 1e-4 y 1e3)
toolbox.register("attr_alpha", np.random.uniform, -4, 3)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_alpha, n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 3 -> definir función de aptitud
def evaluar(individuo):
    log_alpha = individuo[0]
    alpha = 10 ** log_alpha
    modelo = Ridge(alpha=alpha)
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    return(rmse,) #DEAP espera una tupla

toolbox.register("evaluate", evaluar)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=1.0)
toolbox.register("select", tools.selBest)

def hill_climbing(toolbox, generations=30, population_size=20):
    pop = toolbox.population(n=population_size)
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    best_rmse_list = []

    for gen in range(generations):
        offspring = []

        for ind in pop:
            vecino = creator.Individual(ind[:])  # copiar
            toolbox.mutate(vecino)
            del vecino.fitness.values
            vecino.fitness.values = toolbox.evaluate(vecino)

            # Greedy: si mejora, acepta
            if vecino.fitness.values[0] < ind.fitness.values[0]:
                offspring.append(vecino)
            else:
                offspring.append(ind)

        pop = offspring
        best = tools.selBest(pop, k=1)[0]
        best_rmse_list.append(best.fitness.values[0])

        print(f"Generación {gen+1}: Mejor RMSE = {best.fitness.values[0]:.4f}")

    return best, best_rmse_list

mejor_ind, curva = hill_climbing(toolbox)

mejor_alpha = 10 ** mejor_ind[0]
print(f"alpha óptimo encontrado: {mejor_alpha:.5f}")


