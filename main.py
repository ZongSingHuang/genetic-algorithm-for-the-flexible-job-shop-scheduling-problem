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
P = 20
run_times = 1
item = -1
table = pd.DataFrame(np.zeros([10, 7]), columns=['nxm', 'To', 'Flex.', 'LB, UB', 'Cm', 'AV(Cm)', 't'])
table['Cm'] = np.inf

for t in range(run_times):
    # item += 1
    # test = benchmark.test()
    # optimizer = GA(fitness=benchmark.fitness, D=test.total_operation*2, G=G, P=P,
    #                 job=test.job, machine=test.machine, operation=test.total_operation,
    #                 table_np=test.table_np, table_pd=test.table_pd)
    # st = time.time()
    # optimizer.opt()
    # ed = time.time()
    # table.loc[item, 'nxm'] = f'{test.job}x{test.machine}'
    # table.loc[item, 'To'] = test.total_operation
    # table.loc[item, 'Flex.'] = test.machines_per_operation
    # table.loc[item, 'LB, UB'] = '12, 12'
    # table.loc[item, 'Cm'] = min(optimizer.F_gbest, table.loc[item, 'Cm']) if table.loc[item, 'Cm'] else table.loc[item, 'Cm']
    # table.loc[item, 'AV(Cm)'] += optimizer.F_gbest
    # table.loc[item, 't'] += ed - st

    item += 1
    mk01 = benchmark.decoding(path=r'BRdata\Mk01.fjs')
    optimizer = GA(fitness=benchmark.fitness, D=mk01.total_operation*2, G=G, P=P,
                    job=mk01.job, machine=mk01.machine, operation=mk01.total_operation,
                    table_np=mk01.table_np, table_pd=mk01.table_pd)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    table.loc[item, 'nxm'] = f'{mk01.job}x{mk01.machine}'
    table.loc[item, 'To'] = mk01.total_operation
    table.loc[item, 'Flex.'] = mk01.machines_per_operation
    table.loc[item, 'LB, UB'] = '36, 42'
    table.loc[item, 'Cm'] = min(optimizer.F_gbest, table.loc[item, 'Cm']) if table.loc[item, 'Cm'] else table.loc[item, 'Cm']
    table.loc[item, 'AV(Cm)'] += optimizer.F_gbest
    table.loc[item, 't'] += ed - st

    print(t+1)
    item = -1

table['AV(Cm)'] = table['AV(Cm)'] / run_times
table['t'] = table['t'] / run_times
table.to_csv('table(GA).csv')
