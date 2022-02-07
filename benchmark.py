# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 11:05:43 2022

@author: zongsing.huang
"""

import time

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
        self.table_pd['O'] = self.table_pd['job'].astype(str) + self.table_pd['operation'].astype(str)


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
        self.table_pd['O'] = self.table_pd['job'].astype(str) + self.table_pd['operation'].astype(str)


def fitness(X, table_np, table_pd):
    D = int(X.shape[1] / 2)
    F = []
    for _, row in enumerate(X):
        MS = row[:D]
        OS = row[D:]
        number_of_machines = table_pd.columns.size - 3
        number_of_jobs = table_pd['job'].unique().size
        pending = get_pending(MS, OS, table_pd)
        theoretical_limit = int(table_pd.iloc[:, :-3].replace(np.inf, 0).max(axis=1).sum())
        fastest_start_time = np.zeros(number_of_jobs)

        SPACE_columns = ['machine', 'job', 'start', 'end', 'length']
        SPACE_data = np.zeros([number_of_machines, len(SPACE_columns)])
        SPACE = pd.DataFrame(data=SPACE_data, columns=SPACE_columns)
        SPACE['machine'] = range(number_of_machines)
        SPACE['job'] = 'idle'
        SPACE['end'] = theoretical_limit - 1
        SPACE['length'] = theoretical_limit

        GANTT = pd.DataFrame(np.zeros([number_of_machines, theoretical_limit]) - 1, dtype=int)

        for _, val in pending.iterrows():
            # 取得待處理的機台、工件、所需時間
            assigned_machine = val['machine']
            assigned_job = val['job']
            processing_time = val['time']

            # 尋找符合條件的位置
            mask1 = SPACE['machine'] == assigned_machine
            mask2 = SPACE['job'] == 'idle'
            mask3 = SPACE['length'] >= processing_time
            tb = np.maximum(SPACE['start'], fastest_start_time[assigned_job])
            mask4 = tb + processing_time - 1 <= SPACE['end']

            spam = SPACE[mask1 & mask2 & mask3 & mask4].reset_index(drop=True).loc[0]

            # 該筆訂單植入至 GANTT
            tb = np.maximum(spam['start'], fastest_start_time[assigned_job])
            GANTT.loc[spam['machine'], tb:tb+processing_time - 1] = assigned_job

            # 更新 fastest_start_time, SPACE
            fastest_start_time[assigned_job] = tb + processing_time
            mask = SPACE['machine'] == assigned_machine
            SPACE.drop(SPACE[mask].index, axis=0, inplace=True)
            SPACE.reset_index(drop=True, inplace=True)
            st, ed, job = -1, -1, None
            for idx, i in enumerate(GANTT.loc[spam['machine']]):
                if i != job and st == -1:
                    st = idx
                    job = i
                elif i != job and st != -1:
                    ed = idx - 1

                if st != -1 and ed != -1:
                    data = {'machine': spam['machine'],
                            'job': job,
                            'start': st,
                            'end': ed,
                            'length': ed - st + 1}
                    SPACE = SPACE.append(data, ignore_index=True)
                    st, ed, job = idx, -1, i

            if st != -1 and ed == -1:
                ed = idx
                job = i
                data = {'machine': spam['machine'],
                        'job': job,
                        'start': st,
                        'end': ed,
                        'length': ed - st + 1}
                SPACE = SPACE.append(data, ignore_index=True)

            SPACE['job'].replace(-1, 'idle', inplace=True)

        F.append(fastest_start_time.max())

    F = np.array(F)

    return F


def get_pending(MS, OS, table_pd):
    spam = pd.DataFrame(OS, columns=['job'])
    spam['operation'] = -1
    for job in set(OS):
        mask = spam['job'] == job
        spam.loc[mask, 'operation'] = range(mask.sum())
    spam['O'] = spam['job'].astype(str) + spam['operation'].astype(str)

    MS = pd.DataFrame(MS, columns=['machine'])
    MS['O'] = spam['O'].sort_values(ignore_index=True)
    spam = pd.merge(MS, spam, how='right')

    processing_time = []
    for i, j in zip(spam['O'], spam['machine']):
        mask = table_pd['O'] == i
        processing_time.append(table_pd.loc[mask, j].values[0])

    spam['time'] = processing_time

    return spam
