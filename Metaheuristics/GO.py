import numpy as np
from Problem.Benchmark.Problem import fitness as f
from Problem.SCP.problem import SCP
import random as rd


def selectID(popsize, current_id, num_select):
    # Función para seleccionar "num_select" índices aleatorios diferentes de "current_id" en el rango [0, popsize).
    # Se utiliza en la fase de aprendizaje.
    # Esta función evita que se seleccionen índices duplicados en el mismo paso de iteración.
    candidates = list(range(popsize))
    candidates.remove(current_id)
    return np.random.choice(candidates, size=num_select, replace=False)

def GO(popsize, dimension, xmax, xmin, MaxFEs, function, iter, x, fitness, type):
    # Función principal del algoritmo Growth Optimizer (GO).
    # Realiza la optimización en un espacio de búsqueda multidimensional.

    if type == False:
        instance = SCP(function)

    if iter == 0:
        x = xmin + (xmax - xmin) * np.random.rand(popsize, dimension)

    gbestX = np.zeros(dimension)
    gbestfitness = np.inf
    gbesthistory = np.zeros(MaxFEs)

    # Inicialización

    FEs = 0

 
    # Evaluación de la población inicial y actualización del mejor valor global (gbest)
    for i in range(popsize):
        if type == False:
            fitness[i] = instance.fitness(x[i])
        else:
            fitness[i] = f(function, x[i])

        #if gbestfitness > fitness[i]:
        #    gbestfitness = fitness[i]
        #    gbestX = x[i, :]
        #gbesthistory[FEs - 1] = gbestfitness
        #print("FEs: %d, fitness error: %e" % (FEs, gbestfitness))
    
    while True:
        # Ordenar la población según la aptitud (de menor a mayor)
        if FEs >= 1:
            break
        FEs += 1
        ind = np.argsort(fitness)
        Best_X = x[ind[0], :]
        

        # Learning phase (Fase de aprendizaje)
        for i in range(popsize):
            # Selección de soluciones "mejor" (Better_X) y "peor" (Worst_X) para la fase de aprendizaje.
            Worst_X = x[ind[np.random.randint(popsize - 4, popsize)], :]
            Better_X = x[ind[np.random.randint(1, 5)], :]

            # Selección de dos índices aleatorios diferentes para usar en el cálculo de distancias.
            random = selectID(popsize, i, 2)
            L1 = random[0]
            L2 = random[1]

            # Cálculo de distancias y factores de aprendizaje.
            D_value1 = Best_X - Better_X
            D_value2 = Best_X - Worst_X
            D_value3 = Better_X - Worst_X
            D_value4 = x[L1, :] - x[L2, :]

            Gap1 = np.linalg.norm(D_value1)
            Gap2 = np.linalg.norm(D_value2)
            Gap3 = np.linalg.norm(D_value3)
            Gap4 = np.linalg.norm(D_value4)

            rate = Gap1 + Gap2 + Gap3 + Gap4
            if rate == 0:
                rate = rd.random()
            LF1 = Gap1 / rate
            LF2 = Gap2 / rate
            LF3 = Gap3 / rate
            LF4 = Gap4 / rate

            pop = popsize - 1
            # Cálculo de un nuevo vector de posición (newx) para la solución candidata actual (i).
            SF = fitness[i] / fitness[ind[pop]]
            KA1 = LF1 * SF * D_value1
            KA2 = LF2 * SF * D_value2
            KA3 = LF3 * SF * D_value3
            KA4 = LF4 * SF * D_value4
            newx = x[i, :] + KA1 + KA2 + KA3 + KA4

            # Clipping (Asegurarse de que la nueva posición esté dentro de los límites).
            newx = np.maximum(newx, xmin)
            newx = np.minimum(newx, xmax)

            # Evaluación de la aptitud de la nueva posición y actualización de la solución si es mejor.
            if type == False:
                newfitness = instance.fitness(newx)
            else:
                newfitness = f(function, newx)

            if fitness[i] > newfitness:
                fitness[i] = newfitness
                x[i, :] = newx
            else:
                # Con una pequeña probabilidad, también se acepta una solución peor para escapar de mínimos locales.
                if np.random.rand() < 0.001:
                    fitness[i] = newfitness
                    x[i, :] = newx

            # Actualización del mejor valor global (gbest) y registro del historial de la mejor aptitud.
            if gbestfitness > fitness[i]:
                gbestfitness = fitness[i]
                gbestX = x[i, :]
            gbesthistory[FEs - 1] = gbestfitness
            #print("FEs: %d, fitness error: %e" % (FEs, gbestfitness))

        # Reflection phase (Fase de reflexión)
        for i in range(popsize):
            # Generación de una nueva posición aleatoria mediante reflexión en algunas variables de decisión.
            newx = x[i, :]
            j = 0
            while j < dimension:
                if np.random.rand() < 0.3:
                    R = x[ind[np.random.randint(5)], :]
                    newx[j] = x[i, j] + (R[j] - x[i, j]) * np.random.rand()
                    # Con una pequeña probabilidad, también se asigna un valor aleatorio en lugar de reflejar.
                    if np.random.rand() < (0.01 + (0.09) * (1 - iter / MaxFEs)):
                        newx[j] = xmin + (xmax - xmin) * np.random.rand()
                j += 1

            # Clipping (Asegurarse de que la nueva posición esté dentro de los límites).
            newx = np.maximum(newx, xmin)
            newx = np.minimum(newx, xmax)

            # Evaluación de la aptitud de la nueva posición y actualización de la solución si es mejor.
            if type == False:
                newfitness = instance.fitness(newx)
            else:
                newfitness = f(function, newx)

            if fitness[i] > newfitness:
                fitness[i] = newfitness
                x[i, :] = newx
            else:
                # Con una pequeña probabilidad, también se acepta una solución peor para escapar de mínimos locales.
                if np.random.rand() < 0.1 and ind[i] != 0:
                    fitness[i] = newfitness
                    x[i, :] = newx


            # Actualización del mejor valor global (gbest) y registro del historial de la mejor aptitud.
            if gbestfitness > fitness[i]:
                gbestfitness = fitness[i]
                gbestX = x[i, :]
            gbesthistory[FEs - 1] = gbestfitness

    # Retornar el mejor valor encontrado (gbestX), su aptitud (gbestfitness) y el historial de aptitud (gbesthistory).
    return np.array(x)

# Ejemplo de uso:
# Supongamos que tienes una función de aptitud "Fitness" y su identificador "FuncId".
# Y deseas utilizar el algoritmo GO para encontrar el mejor valor en un espacio de búsqueda:
# best_solution, best_fitness, fitness_history = GO(popsize, dimension, xmax, xmin, MaxFEs, Fitness, FuncId)