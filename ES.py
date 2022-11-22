import random
from statistics import mean
from test_functions import *


def initialization(n, mu, max):
    individuals = np.random.uniform(low=-1 * max, high=max, size=(mu, n))
    return individuals


def discrete_recombination(individuals, landa, number=2):
    offsprings = []
    for l in range(landa):
        parents = individuals[np.random.choice(individuals.shape[0], number, replace=False), :]
        offspring = np.zeros((individuals.shape[1]))
        for i in range(offspring.size):
            alleles = [parent[i] for parent in parents]
            offspring[i] = random.choice(alleles)
        offsprings.append(offspring)
    offsprings = np.array(offsprings)
    return offsprings


def intermediary_recombination(individuals, landa, number=2):
    offsprings = []
    for l in range(landa):
        parents = individuals[np.random.choice(individuals.shape[0], number, replace=False), :]
        offspring = np.zeros((individuals.shape[1]))
        for i in range(offspring.size):
            alleles = [parent[i] for parent in parents]
            offspring[i] = mean(alleles)
        offsprings.append(offspring)
    offsprings = np.array(offsprings)
    return offsprings


def adaptive_sigma(sigma, c, ps):
    if ps == 1 / 5:
        return sigma
    if ps > 1 / 5:
        return sigma / c
    if ps < 1 / 5:
        return sigma * c
    return sigma


def compute_successful_mutation(offsprings, mutated_offsprings, function):
    offspring_fitness = np.apply_along_axis(function, axis=1, arr=offsprings)
    mutated_offsprings_fitness = np.apply_along_axis(function, axis=1, arr=mutated_offsprings)
    return np.count_nonzero(np.greater(mutated_offsprings_fitness, offspring_fitness))


n = 20
c = 0.9
k = 5
sigma = 0.01
landa = 1000
mu = 100

############################################################ Griewank
function = griewank
max_range = 100

############################################################ Rastrigin
# function = rastrigin
# max_range = 5.12

############################################################ Schwefel
# function = schwefel
# max_range = 500


gen = 0
gens = []
best_mins = []
current_k = 0
total_mutation = 0
successful_mutation = 0
individuals = initialization(n=n, mu=mu, max=max_range)
while gen < 100:
    gen = gen + 1

    # offsprings = discrete_recombination(individuals=individuals, landa=landa)
    offsprings = intermediary_recombination(individuals=individuals, landa=landa)

    current_k = current_k + 1
    if current_k % k == 0:
        ps = successful_mutation / total_mutation
        sigma = adaptive_sigma(sigma, c, ps)

    mutated_offsprings = offsprings + sigma
    total_mutation = total_mutation + offsprings.shape[0]
    successful_mutation = successful_mutation + compute_successful_mutation(offsprings, mutated_offsprings, function)

    ############################################################    mu + landa
    # all_ind_offspring = np.concatenate((individuals, mutated_offsprings), axis=0)
    # all_ind_offspring_fitness = np.apply_along_axis(function, axis=1, arr=all_ind_offspring)
    # best_indices = np.argsort(all_ind_offspring_fitness)[:mu]
    # new_generation = all_ind_offspring[best_indices, :]

    ############################################################     mu, landa
    all_offspring_fitness = np.apply_along_axis(function, axis=1, arr=mutated_offsprings)
    best_indices = np.argsort(all_offspring_fitness)[:mu]
    new_generation = mutated_offsprings[best_indices, :]

    individuals = new_generation

    gens.append(gen)
    print(f'Generation = {gen}')
    best_mins.append(np.min(np.apply_along_axis(function, axis=1, arr=individuals)))

print(f'best={best_mins[-1]}')
plt.plot(gens, best_mins)
plt.title(label=f'ES - best={str(best_mins[-1])}')
plt.xlabel('generation')
plt.ylabel('fitness')
plt.savefig('ES.png')
plt.show()
