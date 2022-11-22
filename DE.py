import random
from test_functions import *


def initialization(n, mu, max):
    individuals = np.random.uniform(low=-1 * max, high=max, size=(mu, n))
    return individuals


def crossover(individuals, mutants):
    trials = np.zeros_like(individuals)
    for g in range(individuals.shape[0]):
        for i in range(individuals.shape[1]):
            random_parent = random.normalvariate(0, 1)
            if random_parent > 0.5:
                trials[g, i] = individuals[g, i]
            else:
                trials[g, i] = mutants[g, i]
        random_position = random.randint(0, individuals.shape[1] - 1)
        trials[g, random_position] = individuals[g, i]
    return trials


def make_mutants(individuals):
    mutants = np.zeros_like(individuals)
    for i in range(individuals.shape[0]):
        choosed = individuals[np.random.choice(individuals.shape[0], 3, replace=False), :]
        mutants[i] = choosed[0] + f * (choosed[1] - choosed[2])
    return mutants


n = 20
mu = 100
# f = np.random.uniform(0, 1)
f = 0.9
############################################################ Griewank
function = griewank
max_range = 100

############################################################ Rastrigin
# function = rastrigin
# max_range = 5.12

############################################################ Schwefel
# function = schwefel
# max_range = 500

individuals = initialization(n=n, mu=mu, max=max_range)

gen = 0
gens = []
best_mins = []
while gen < 100:
    gen = gen + 1
    mutants = make_mutants(individuals=individuals)
    trials = crossover(individuals, mutants)

    individuals_fitness = np.apply_along_axis(function, axis=1, arr=individuals)
    trials_fitness = np.apply_along_axis(function, axis=1, arr=trials)

    next_generation = np.zeros_like(individuals)
    for i in range(individuals.shape[1]):
        if individuals_fitness[i] < trials_fitness[i]:
            next_generation[i] = individuals[i]
        else:
            next_generation[i] = trials[i]

    individuals = next_generation
    best_mins.append(np.mean(np.apply_along_axis(function, axis=1, arr=individuals)))
    print(f'Generation = {gen}')
    gens.append(gen)

print(f'best={best_mins[-1]}')
plt.plot(gens, best_mins)
plt.title(label=f'DE - f={f:.2f} - best={str(best_mins[-1])}')
plt.xlabel('generation')
plt.ylabel('fitness')
plt.savefig(f'DE.png')
plt.show()
