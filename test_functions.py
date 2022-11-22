import math
import numpy as np
import matplotlib.pyplot as plt


def rastrigin(x):
    sum = 0
    for g in x:
        sum += (g ** 2.0) - 10 * math.cos(2 * math.pi * g)
    n = float(len(x))
    return 10 * n + sum


def schwefel(x):
    sum = 0
    for g in x:
        sum += -1 * g * math.sin(math.sqrt(math.fabs(g)))
    n = float(len(x))
    return 418.9829 * n + sum


def griewank(x):
    sum = 1
    mul = 1
    for g in x:
        sum += g ** 2 / 4000
        mul *= math.cos(g / math.sqrt(np.where(x == g)[0][0] + 1))
    return sum - mul


############################################################ plot functions
# x = np.arange(-500, 500, 0.01)
# y = list()
# for g in list(x):
#     y.append(schwefel([g]))
# plt.plot(x, y)
# plt.savefig('schwefel.png')
# plt.show()
