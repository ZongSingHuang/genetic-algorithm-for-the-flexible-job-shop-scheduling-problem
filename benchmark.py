# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 11:05:43 2022

@author: zongsing.huang
"""

import numpy as np
import pandas as pd


class test:
    def __init__(self):
        self.job = 2
        self.machine = 5
        self.machines_per_operation = 666
        self.total_operation = 5

        # [工件, 製程]
        self.reference = np.array([[0, 0],
                                   [0, 1],
                                   [1, 0],
                                   [1, 1],
                                   [1, 2]])

        self.table_np = np.array([[2,      6,      5,      3,      4],
                                  [np.inf, 8,      np.inf, 4,      np.inf],
                                  [3,      np.inf, 6,      np.inf, 5],
                                  [4,      6,      5,      np.inf, np.inf],
                                  [np.inf, 7,      11,     5,      8]])
        self.table_np = np.hstack([self.table_np, self.reference])

        self.table_pd = pd.DataFrame(data=self.table_np)
        self.table_pd.rename(columns={self.table_pd.columns[-2]: 'job',
                                      self.table_pd.columns[-1]: 'operation'},
                             inplace=True)
        self.table_pd['job'] = self.table_pd['job'].astype(int)
        self.table_pd['operation'] = self.table_pd['operation'].astype(int)


class decoding:
    def __init__(self, path):
        self.table_raw = pd.read_table(path, header=None)
        self.job = int(self.table_raw.loc[0, 0])
        self.machine = int(self.table_raw.loc[0, 1])
        self.machines_per_operation = self.table_raw.loc[0, 2]
        self.table_ok = self.table_raw.loc[1:, 0].str.split(expand=True).astype(float)
        self.total_operation = int(self.table_ok[0].sum())

        # [工件, 製程]
        self.reference = []
        for i in range(self.job):
            for j in range(self.table_ok[0].astype(int).tolist()[i]):
                self.reference.append([int(i), int(j)])
        self.reference = np.array(self.reference)
        self.table_ok.drop([0], axis=1, inplace=True)

        spam = []
        for idx, row in self.table_ok.iterrows():
            row.dropna(inplace=True)
            row = row.astype(int).tolist()
            while row:
                number_of_devices_available = row[0]
                spam.append(row[1:1+number_of_devices_available*2])
                row = row[1+number_of_devices_available*2:]

        self.table_np = np.zeros([self.total_operation, self.machine]) + np.inf
        for operation_id, row in enumerate(spam):
            row = np.array(row).reshape(-1, 2)
            row[:, 0] -= 1
            machine_id = row[:, 0]
            process_time = row[:, 1]
            self.table_np[operation_id, machine_id] = process_time

        self.table_np = np.hstack([self.table_np, self.reference])

        self.table_pd = pd.DataFrame(data=self.table_np)
        self.table_pd.rename(columns={self.table_pd.columns[-2]: 'job',
                                      self.table_pd.columns[-1]: 'operation'},
                             inplace=True)
        self.table_pd['job'] = self.table_pd['job'].astype(int)
        self.table_pd['operation'] = self.table_pd['operation'].astype(int)


def fitness(X, table_np, table_pd):
    table_pd.set_index(table_pd['job'].astype(str) + table_pd['operation'].astype(str), inplace=True)
    D = int(X.shape[1] / 2)
    for idx, row in enumerate(X):
        MS = row[:D]
        OS = row[D:]
        summary = get_summary(MS, OS, table_pd)
        print(row)
    P = len(X)
    return np.random.uniform(size=P)


def get_summary(MS, OS, table_pd):
    spam = pd.DataFrame(OS, columns=['job'])
    spam['operation'] = -1
    for job in set(OS):
        mask = spam['job'] == job
        spam.loc[mask, 'operation'] = range(mask.sum())
    spam['O'] = spam['job'].astype(str) + spam['operation'].astype(str)

    spam[['time', 'machine']] = [(table_pd.loc[i, j], j) for i, j in zip(spam['O'], MS)]
    spam[['job', 'operation', 'machine']] = spam[['job', 'operation', 'machine']].astype(int)

    return spam
