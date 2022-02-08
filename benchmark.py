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

        SPACE_columns = ['machine', 'job', 'start', 'end', 'length', 'O']
        SPACE_data = np.zeros([number_of_machines, len(SPACE_columns)])
        SPACE = pd.DataFrame(data=SPACE_data, columns=SPACE_columns)
        SPACE['machine'] = range(number_of_machines)
        SPACE['job'] = 'idle'
        SPACE['end'] = theoretical_limit - 1
        SPACE['length'] = theoretical_limit
        SPACE['O'] = 'idle'

          # 為了提速所以關掉
        # GANTT = pd.DataFrame('idle', index=range(number_of_machines), columns=range(theoretical_limit))

        pending = pending.to_dict('records')
        for val in pending:
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
            available_space = SPACE[mask1 & mask2 & mask3 & mask4].reset_index(drop=True).loc[0]
            available_space_idx = SPACE[mask1 & mask2 & mask3 & mask4].index[0]
            SPACE = SPACE.to_dict('records')
            # 該筆訂單植入至 GANTT
            tb = np.maximum(available_space['start'], fastest_start_time[assigned_job])
            # 為了提速所以關掉
            # if not all(GANTT.loc[available_space['machine'], tb:tb+processing_time - 1].values == 'idle'):
            #     print('cover!!!!')
            # GANTT.loc[available_space['machine'], tb:tb+processing_time - 1] = assigned_job

            # 更新 SPACE
            # case 1
            if available_space['length'] == processing_time:
                data1 = {'machine': available_space['machine'],
                         'job': assigned_job,
                         'start': available_space['start'],
                         'end': available_space['end'],
                         'length': processing_time,
                         'O': val['O']}
                SPACE.append(data1)

            # case 2
            elif available_space['start'] >= fastest_start_time[assigned_job]:
                data1 = {'machine': available_space['machine'],
                         'job': assigned_job,
                         'start': tb,
                         'end': tb + processing_time - 1,
                         'length': processing_time,
                         'O': val['O']}

                data2 = {'machine': available_space['machine'],
                         'job': 'idle',
                         'start': tb + processing_time,
                         'end': available_space['end'],
                         'length': available_space['length'] - processing_time,
                         'O': 'idle'}

                SPACE.append(data1)
                SPACE.append(data2)

            # case 3
            elif available_space['end'] == tb + processing_time - 1:
                data1 = {'machine': available_space['machine'],
                         'job': 'idle',
                         'start': available_space['start'],
                         'end': available_space['end'] - processing_time,
                         'length': available_space['length'] - processing_time,
                         'O': 'idle'}

                data2 = {'machine': available_space['machine'],
                         'job': assigned_job,
                         'start': tb + processing_time - 1,
                         'end': available_space['end'],
                         'length': processing_time,
                         'O': val['O']}

                SPACE.append(data1)
                SPACE.append(data2)

            # case 4
            else:
                data1 = {'machine': available_space['machine'],
                         'job': 'idle',
                         'start': available_space['start'],
                         'end': tb - 1,
                         'length': tb - available_space['start'],
                         'O': 'idle'}

                data2 = {'machine': available_space['machine'],
                         'job': assigned_job,
                         'start': tb,
                         'end': tb + processing_time - 1,
                         'length': processing_time,
                         'O': val['O']}

                data3 = {'machine': available_space['machine'],
                         'job': 'idle',
                         'start': tb + processing_time,
                         'end': available_space['end'],
                         'length': available_space['end'] - (tb + processing_time) + 1,
                         'O': 'idle'}

                SPACE.append(data1)
                SPACE.append(data2)
                SPACE.append(data3)

            del SPACE[available_space_idx]
            SPACE = pd.DataFrame(SPACE)
            # 為了提速所以關掉
            # SPACE.sort_values(['machine', 'start'], inplace=True)

            # 更新 fastest_start_time
            fastest_start_time[assigned_job] = tb + processing_time

        F.append(fastest_start_time.max())

    F = np.array(F)

    return F


def get_pending(MS, OS, table_pd):
    pending = table_pd.copy()
    pending['machine'] = MS
    pending.set_index('O', drop=False, inplace=True)

    spam = pd.DataFrame(OS, columns=['job'])
    for job in set(OS):
        mask = spam['job'] == job
        spam.loc[mask, 'operation'] = range(mask.sum())
    spam['operation'] = spam['operation'].astype(int)
    spam['O'] = spam['job'].astype(str) + spam['operation'].astype(str)

    pending = pending.loc[spam['O']]
    pending['time'] = [pending.loc[i, j] for i, j in zip(pending['O'], pending['machine'])]

    pending = pending[['machine', 'O', 'job', 'operation', 'time']].reset_index(drop=True)

    return pending
