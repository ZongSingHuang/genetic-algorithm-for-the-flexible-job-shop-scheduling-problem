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
item = -1
table = pd.DataFrame(np.zeros([10, 7]), columns=['nxm', 'To', 'Flex.', 'LB, UB', 'Cm', 'AV(Cm)', 't'])

for t in range(run_times):
    item += 1
    test = benchmark.test()
    # optimizer = GA(fitness=benchmark.Sphere,
    #                D=test.total_operation * 2, P=P, G=G)
    st = time.time()
    # optimizer.opt()
    ed = time.time()
    table.loc[item, 'nxm'] = f'{test.job}x{test.machine}'
    table.loc[item, 'To'] = test.total_operation
    table.loc[item, 'Flex.'] = test.machines_per_operation
    table.loc[item, 'LB, UB'] = '?, ?'
    # table.loc[item, 'Cm'] = min(optimizer.F_gbest, table.loc[item, 'Cm']) if table.loc[item, 'Cm'] else table.loc[item, 'Cm']
    # table.loc[item, 'AV(Cm)'] += table.loc[item, 'Cm']
    # table.loc[item, 't'] += ed - st

    item += 1
    mk01 = benchmark.decoding(path=r'BRdata\Mk01.fjs')
    # optimizer = GA(fitness=benchmark.Sphere,
    #                D=mk01.total_operation * 2, P=P, G=G)
    st = time.time()
    # optimizer.opt()
    ed = time.time()
    table.loc[item, 'nxm'] = f'{mk01.job}x{mk01.machine}'
    table.loc[item, 'To'] = mk01.total_operation
    table.loc[item, 'Flex.'] = mk01.machines_per_operation
    table.loc[item, 'LB, UB'] = '36, 42'
    # table.loc[item, 'Cm'] = min(optimizer.F_gbest, table.loc[item, 'Cm']) if table.loc[item, 'Cm'] else table.loc[item, 'Cm']
    # table.loc[item, 'AV(Cm)'] += table.loc[item, 'Cm']
    # table.loc[item, 't'] += ed - st

    # print(t+1)

table['AV(Cm)'] = table['AV(Cm)'] / run_times
table['t'] = table['t'] / run_times
table.to_csv('table(GA).csv')
