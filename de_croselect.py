"""""
   Filename: de_croselect.py
Description: The (DE/current-to-pbest) mutation and (binomial) crossover operations in differential evolution.
      Paper: J. Sun, X. Liu, T. Bäck and Z. Xu, "Learning Adaptive Differential Evolution Algorithm From
             Optimization Experiences by Policy Gradient," in IEEE Transactions on Evolutionary Computation,
             vol. 25, no. 4, pp. 666-680, Aug. 2021, doi: 10.1109/TEVC.2021.3060811.
     Author: Xin Liu
"""""
from globalVar import *
from CEC13 import *
from CEC17 import *


def modifyChildwithParent(cross_pop, parent_pop, x_max, x_min):
    for i in range(cross_pop.shape[0]):
        for j in range(cross_pop.shape[1]):
            if cross_pop[i, j] < x_min:
                cross_pop[i, j] = (parent_pop[i, j] + x_min)/2.0
            elif cross_pop[i, j] > x_max:
                cross_pop[i, j] = (parent_pop[i, j] + x_max)/2.0
    return cross_pop


def de_crosselect(pop, m_pop, fit, cr_vector, nfes, index_func, train_flag):
    n_pop = np.zeros_like(pop)
    n_fit = np.zeros_like(fit)
    cr = np.random.uniform(size=(POP_SIZE, PROBLEM_SIZE))
    cr_p = cr.copy()
    for i in range(POP_SIZE):
        cr_p[i, cr_p[i, :] < cr_vector[0, i]] = 0
        cr_p[i, cr_p[i, :] >= cr_vector[0, i]] = 1
    cr_m = (cr_p == False).astype('float')
    for i in range(POP_SIZE):
        j = np.random.randint(PROBLEM_SIZE)
        cr_m[i, j] = 1
    cr_p = (cr_m == False).astype('float')
    cross_pop = cr_p * pop + cr_m * m_pop

    cross_pop = modifyChildwithParent(cross_pop, pop, X_MAX, X_MIN)
    if train_flag == 1:
        cross_fit = cec13(cross_pop, index_func)
    else:
        cross_fit = cec17(cross_pop, index_func)

    nfes += POP_SIZE
    for i in range(POP_SIZE):
        if cross_fit[i] <= fit[i]:
            n_fit[i] = cross_fit[i]
            n_pop[i] = cross_pop[i]
        else:
            n_fit[i] = fit[i]
            n_pop[i] = pop[i]
    return n_pop, n_fit, nfes
