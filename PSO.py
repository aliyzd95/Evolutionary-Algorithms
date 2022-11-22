import random
from statistics import mean
from test_functions import *


def initialization(n, mu, max):
    individuals = np.random.uniform(low=-1 * max, high=max, size=(mu, n))
    return individuals


def update_velocities(individuals, velocities, bests, whole_best, w, phi1, phi2):
    whole_best = whole_best.reshape(-1, 1).T
    whole_best1 = whole_best
    for i in range(individuals.shape[0] - 1):
        whole_best = np.concatenate((whole_best, whole_best1), axis=0)
    u1 = np.random.uniform(low=0, high=1, size=(individuals.shape[0], individuals.shape[1]))
    u2 = np.random.uniform(low=0, high=1, size=(individuals.shape[0], individuals.shape[1]))
    velocities = w * velocities + phi1 * np.multiply(u1, bests - individuals) + phi2 * np.multiply(u2,
                                                                                                   whole_best - individuals)
    return velocities


def update_bests(individuals, bests, individuals_fitness, bests_fitness):
    for i in range(individuals_fitness.shape[0]):
        if individuals_fitness[i] < bests_fitness[i]:
            bests[i] = individuals[i]
    return bests


n = 20
mu = 100
w = 0.8
# phi1 = np.random.uniform(0, 3)
# phi2 = np.random.uniform(0, 3)
phi1 = 0.01
phi2 = 0.1

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
velocities = np.zeros((mu, n))
bests = individuals
whole_best = bests[np.argmin(np.apply_along_axis(function, axis=1, arr=bests))]

gen = 0
gens = []
best_mins = []
while gen < 100:
    gen = gen + 1

    velocities = update_velocities(individuals, velocities, bests, whole_best, w, phi1, phi2)
    individuals = np.add(individuals, velocities)

    individuals_fitness = np.apply_along_axis(function, axis=1, arr=individuals)
    bests_fitness = np.apply_along_axis(function, axis=1, arr=bests)
    bests = update_bests(individuals, bests, individuals_fitness, bests_fitness)

    whole_best = bests[np.argmin(bests_fitness)]

    gens.append(gen)
    print(f'Generation = {gen}')
    best_mins.append(np.min(np.apply_along_axis(function, axis=1, arr=individuals)))

print(f'best={best_mins[-1]}')
plt.plot(gens, best_mins)
plt.title(label=f'PSO - W=0.8 (phi1={phi1:.2f}, phi2={phi2:.2f}) - best={str(best_mins[-1])}')
plt.xlabel('generation')
plt.ylabel('fitness')
plt.savefig('PSO.png')
plt.show()
