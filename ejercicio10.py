import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from deap import base, creator, tools
import random
import copy
import matplotlib.pyplot as plt

# Cargar datos
df = pd.read_csv("dataset10.csv", sep=";")
X = df[['Credits', 'Prev_GPA', 'Extracurricular_hours']].values
y = LabelEncoder().fit_transform(df['Category'])

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

# DEAP Setup
if "FitnessMax" not in creator.__dict__:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if "Individual" not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMax)

def generar_individuo():
    return [
        random.randint(1, 3),  # capas
        random.randint(4, 64),  # neuronas capa 1
        random.randint(4, 64),  # capa 2
        random.randint(4, 64),  # capa 3
        random.uniform(0.0001, 0.1)  # learning rate
    ]

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, generar_individuo)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Red neuronal en PyTorch
class Net(nn.Module):
    def __init__(self, layers, neurons, n_inputs=3, n_outputs=3):
        super().__init__()
        self.hidden = nn.ModuleList()
        self.hidden.append(nn.Linear(n_inputs, neurons[0]))
        for i in range(1, layers):
            self.hidden.append(nn.Linear(neurons[i - 1], neurons[i]))
        self.output = nn.Linear(neurons[layers - 1], n_outputs)

    def forward(self, x):
        for layer in self.hidden:
            x = torch.relu(layer(x))
        return self.output(x)

# Evaluación
def evaluar_individuo(ind):
    layers = ind[0]
    neurons = ind[1:4]
    lr = ind[4]

    model = Net(layers, neurons)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for _ in range(20):  # epochs = 20
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pred = model(X_test_t).argmax(dim=1)
        acc = accuracy_score(y_test, pred.numpy())
    return (acc,)

toolbox.register("evaluate", evaluar_individuo)
toolbox.register("select", tools.selTournament, tournsize=3)

def mutar(ind):
    if random.random() < 0.3:
        ind[0] = random.randint(1, 3)
    for i in range(1, 4):
        if random.random() < 0.5:
            ind[i] = max(4, min(64, ind[i] + random.randint(-8, 8)))
    if random.random() < 0.5:
        ind[4] = max(0.0001, min(0.1, ind[4] + random.uniform(-0.01, 0.01)))
    return ind,

toolbox.register("mutate", mutar)

def hill_climbing(ind, steps=5):
    best = copy.deepcopy(ind)
    best_acc = evaluar_individuo(best)[0]
    for _ in range(steps):
        neighbor = mutar(copy.deepcopy(best))[0]
        acc = evaluar_individuo(neighbor)[0]
        if acc > best_acc:
            best = neighbor
            best_acc = acc
    return best

def evolucion_red(ngen=20, pop_size=20):
    pop = toolbox.population(n=pop_size)
    accuracies = []

    for gen in range(ngen):
        offspring = []
        for ind in pop:
            child = toolbox.clone(ind)
            toolbox.mutate(child)
            child = hill_climbing(child)
            child.fitness.values = toolbox.evaluate(child)
            offspring.append(child)

        pop = toolbox.select(offspring, k=pop_size)
        best = tools.selBest(pop, k=1)[0]
        accuracies.append(best.fitness.values[0])
        print(f"Gen {gen}: Accuracy = {best.fitness.values[0]:.4f}")

    best = tools.selBest(pop, k=1)[0]
    return best, accuracies

def graficar_accuracy(accuracies):
    plt.plot(accuracies, marker='o')
    plt.title("Accuracy por generación")
    plt.xlabel("Generación")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.show()

# Ejecutar evolución
mejor_ind, accs = evolucion_red()
graficar_accuracy(accs)

print("Mejor arquitectura:", mejor_ind)
