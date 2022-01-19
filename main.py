# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 21:43:03 2020

@author: ZongSing_NB
"""

import time
import functools

import numpy as np
import pandas as pd

from GA import GA
import benchmark

G = 100
P = 50
run_times = 5
table = pd.DataFrame(np.zeros([6, 36]), index=['avg', 'std', 'worst', 'best', 'ideal', 'time'])
loss_curves = np.zeros([G, 36])
F_table = np.zeros([run_times, 36])

for t in range(run_times):
    item = 0
    mk01 = benchmark.mk01(path=r'BRdata\Mk01.fjs')
    # optimizer = GA(fitness=benchmark.Sphere,
    #                D=mk01.total_operation, P=P, G=G)
    st = time.time()
    # optimizer.opt()
    ed = time.time()
    table[item]['n'] = mk01.job
    table[item]['m'] = mk01.machine
    table[item]['To'] = mk01.total_operation
    table[item]['Flex.'] = mk01.machines_per_operation
    table[item]['LB'] = mk01.optimum_makespan
    # table[item]['Cm'] = min(optimizer.gbest_F, table[item]['Cm'])
    # table[item]['AV(Cm)'] += optimizer.F_gbest
    # table[item]['t'] += optimizer.F_gbest

    # print(t+1)

# loss_curves = loss_curves / run_times
# loss_curves = pd.DataFrame(loss_curves)
# loss_curves.columns = ['Sphere', 'Rastrigin', 'Ackley', 'Griewank', 'Schwefel P2.22',
#                        'Rosenbrock', 'Sehwwefel P2.21', 'Quartic', 'Schwefel P1.2', 'Penalized 1',
#                        'Penalized 2', 'Schwefel P2.26', 'Step', 'Kowalik', 'Shekel Foxholes',
#                        'Goldstein-Price', 'Shekel 5', 'Branin', 'Hartmann 3', 'Shekel 7',
#                        'Shekel 10', 'Six-Hump Camel-Back', 'Hartmann 6', 'Zakharov', 'Sum Squares',
#                        'Alpine', 'Michalewicz', 'Exponential', 'Schaffer', 'Bent Cigar',
#                        'Bohachevsky 1', 'Elliptic', 'Drop Wave', 'Cosine Mixture', 'Ellipsoidal',
#                        'Levy and Montalvo 1']
# loss_curves.to_csv('loss_curves(PSO).csv')

# table.loc[['avg', 'time']] = table.loc[['avg', 'time']] / run_times
# table.loc['worst'] = F_table.max(axis=0)
# table.loc['best'] = F_table.min(axis=0)
# table.loc['std'] = F_table.std(axis=0)
# table.columns = ['Sphere', 'Rastrigin', 'Ackley', 'Griewank', 'Schwefel P2.22',
#                  'Rosenbrock', 'Sehwwefel P2.21', 'Quartic', 'Schwefel P1.2', 'Penalized 1',
#                  'Penalized 2', 'Schwefel P2.26', 'Step', 'Kowalik', 'Shekel Foxholes',
#                  'Goldstein-Price', 'Shekel 5', 'Branin', 'Hartmann 3', 'Shekel 7',
#                  'Shekel 10', 'Six-Hump Camel-Back', 'Hartmann 6', 'Zakharov', 'Sum Squares',
#                  'Alpine', 'Michalewicz', 'Exponential', 'Schaffer', 'Bent Cigar',
#                  'Bohachevsky 1', 'Elliptic', 'Drop Wave', 'Cosine Mixture', 'Ellipsoidal',
#                  'Levy and Montalvo 1']
# table.to_csv('table(PSO).csv')