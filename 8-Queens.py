import numpy as np
import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose

fitness = mlrose.Queens()

def queens_max(state):
    
    fitness = 0 

    for i in range(len(state)-1):
        for j in range(i+1, len(state)):

            if (state[j] != state[i])\
                and (state[j] != state[i]+(j-i))\
                and (state[j] != state[i]-(j-i)):
                    fitness += 1
    return fitness 

fitness_cust = mlrose.CustomFitness(queens_max)

problem = mlrose.DiscreteOpt(length=8, fitness_fn=fitness, 
                            maximize=False, max_val=8)

schedule = mlrose.ExpDecay()
init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7])

best_state, best_fitness = mlrose.simulated_annealing(problem, schedule=schedule,
                                                     max_attempts=10, max_iters=1000,
                                                     init_state=init_state, random_state=1)

print('the best state found is: ', best_state)
print('the fitness at the best state is: ', best_fitness)