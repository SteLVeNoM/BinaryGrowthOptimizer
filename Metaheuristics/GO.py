import numpy as np
from Problem.SCP.problem import SCP
import random as rd

def selectID(popsize, current_id, num_select):
    # Función para seleccionar "num_select" índices aleatorios diferentes de "current_id" en el rango [0, popsize).
    # Se utiliza en la fase de aprendizaje.
    # Esta función evita que se seleccionen índices duplicados en el mismo paso de iteración.
    candidates = list(range(popsize))
    candidates.remove(current_id)
    return np.random.choice(candidates, size=num_select, replace=False)

### 1) SE CAMBIA LAS ENTRADAS DE FITNESS Y FUNCID POR INSTANCIA, SE IMPORTA BASE DE DATOS Y CLASS SCP
### 2) SE ELIMINA LA FASE DE INICIALIZACIÓN DE GO, SE AGREGAN VARIABLES DE ENTRADAS MAXITER Y POBLACIÓN
### Y UTILIZA LA INICIALIZACIÓN E ITERACIÓN DEL PROGRAMA DE PROFESOR/FELIPE (MAXFES = 1)

def GO(popsize, dimension, xmax, xmin, MaxFEs, instancia, FEs, maxIter, x):
    # Función principal del algoritmo Growth Optimizer (GO).
    # Realiza la optimización en un espacio de búsqueda multidimensional.
    instance = SCP(instancia)
    # Inicialización
    #FEs = 0
    #x = xmin + (xmax - xmin) * np.random.rand(popsize, dimension)
    
### 3) SE ELIMINAN LAS VARIABLES GBESTFITNESS, GBESTHISTORY Y GBESTX YA QUE EL PROGRAMA DE PROFESOR/FELIPE LLEVA 
### REGISTRO DE LAS MEJORES SOLUCIONES E HISTORIAL DE MEJORES SOLUCIONES.


    #gbestfitness = np.inf
    fitness = np.zeros(popsize)
    #gbestX = np.zeros(dimension)
    #gbesthistory = np.zeros(MaxFEs)

    # Evaluación de la población inicial y actualización del mejor valor global (gbest)
    for i in range(popsize):
        fitness[i] = instance.fitness(x[i])
        #if gbestfitness > fitness[i]:
        #    gbestfitness = fitness[i]
        #    gbestX = x[i, :]
        #gbesthistory[FEs - 1] = gbestfitness
        #print("FEs: %d, fitness error: %e" % (FEs, gbestfitness))
    
    FE1s = 0
    while True:
        # Ordenar la población según la aptitud (de menor a mayor)
        ind = np.argsort(fitness)
        Best_X = x[ind[0], :]
        

        # Learning phase (Fase de aprendizaje)
        for i in range(popsize):
            # Selección de soluciones "mejor" (Better_X) y "peor" (Worst_X) para la fase de aprendizaje.
            Worst_X = x[ind[np.random.randint(popsize - 4, popsize)], :]
            Better_X = x[ind[np.random.randint(2, 5)], :]

            # Selección de dos índices aleatorios diferentes para usar en el cálculo de distancias.
            random = selectID(popsize, i, 2)
            L1 = random[0]
            L2 = random[1]

            # Cálculo de distancias y factores de aprendizaje.
            D_value1 = Best_X - Better_X
            D_value2 = Best_X - Worst_X
            D_value3 = Better_X - Worst_X
            D_value4 = x[L1, :] - x[L2, :]

            Distance1 = np.linalg.norm(D_value1)
            Distance2 = np.linalg.norm(D_value2)
            Distance3 = np.linalg.norm(D_value3)
            Distance4 = np.linalg.norm(D_value4)

            rate = Distance1 + Distance2 + Distance3 + Distance4
            
            #AGREGUE PARA EVITAR ERROR DE DESBORDAMIENTOS
            if rate == 0: 
                rate = rd.randint(1,5)
            
            LF1 = Distance1 / rate
            LF2 = Distance2 / rate
            LF3 = Distance3 / rate
            LF4 = Distance4 / rate

            # Cálculo de un nuevo vector de posición (newx) para la solución candidata actual (i).
            SF = fitness[i] / np.max(fitness)
            Gap1 = LF1 * SF * D_value1
            Gap2 = LF2 * SF * D_value2
            Gap3 = LF3 * SF * D_value3
            Gap4 = LF4 * SF * D_value4
            newx = x[i, :] + Gap1 + Gap2 + Gap3 + Gap4

            # Clipping (Asegurarse de que la nueva posición esté dentro de los límites).
            newx = np.maximum(newx, xmin)
            newx = np.minimum(newx, xmax)

            # Evaluación de la aptitud de la nueva posición y actualización de la solución si es mejor.
            newfitness = instance.fitness(newx)
            if fitness[i] > newfitness:
                fitness[i] = newfitness
                x[i, :] = newx
            else:
                # Con una pequeña probabilidad, también se acepta una solución peor para escapar de mínimos locales.
                if np.random.rand() < 0.01 and ind[i] != 0:
                    fitness[i] = newfitness
                    x[i, :] = newx

            # Actualización del mejor valor global (gbest) y registro del historial de la mejor aptitud.
            # if gbestfitness > fitness[i]:
            #    gbestfitness = fitness[i]
            #    gbestX = x[i, :]
            #gbesthistory[FEs - 1] = gbestfitness
            #print("FEs: %d, fitness error: %e" % (FEs, gbestfitness))

        if FE1s >= MaxFEs:
            break
      

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
                    if np.random.rand() < (0.01 + (0.99) * (1 - FEs / maxIter)):
                        newx[j] = xmin + (xmax - xmin) * np.random.rand() 
                j += 1

            # Clipping (Asegurarse de que la nueva posición esté dentro de los límites).
            newx = np.maximum(newx, xmin)
            newx = np.minimum(newx, xmax)

            # Evaluación de la aptitud de la nueva posición y actualización de la solución si es mejor.
            newfitness = instance.fitness(newx)
            if fitness[i] > newfitness:
                fitness[i] = newfitness
                x[i, :] = newx
            else:
                # Con una pequeña probabilidad, también se acepta una solución peor para escapar de mínimos locales.
                if np.random.rand() < 0.001 and ind[i] != 0:
                    fitness[i] = newfitness
                    x[i, :] = newx

            # Actualización del mejor valor global (gbest) y registro del historial de la mejor aptitud.
            # if gbestfitness > fitness[i]:
            #    gbestfitness = fitness[i]
            #    gbestX = x[i, :]
            #gbesthistory[FEs - 1] = gbestfitness
        FE1s +=1

    
    # Retornar el mejor valor encontrado (gbestX), su aptitud (gbestfitness) y el historial de aptitud (gbesthistory).
    return np.array(x) #,gbestfitness, gbesthistory