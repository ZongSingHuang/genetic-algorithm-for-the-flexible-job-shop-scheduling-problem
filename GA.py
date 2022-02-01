# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 16:13:49 2020

@author: e10832
"""
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)


class GA():
    def __init__(self, fitness, D, P, job, machine, operation, table_np, table_pd,
                 G=100, GS=0.6, LS=0.3, RS=0.1, pc=0.7, pTPX=0.5, pUX=0.5, pm=0.01):
        self.fitness = fitness
        self.D = D
        self.P = P
        self.G = G
        self.job = job
        self.machine = machine
        self.operation = operation
        self.GS = GS
        self.LS = LS
        self.RS = RS
        self.pc = pc
        self.pTPX = pTPX
        self.pUX = pUX
        self.pm = pm
        self.table_np = table_np
        self.table_pd = table_pd

        self.X_pbest = np.zeros([self.P, self.D])
        self.F_pbest = np.zeros([self.P]) + np.inf
        self.X_gbest = np.zeros([self.D])
        self.F_gbest = np.inf
        self.loss_curve = np.zeros(self.G)

    def opt(self):
        # 初始化
        X1 = []

        # 初始化 : MS
        P_gs = int(self.P * self.GS)
        for idx_chromosome in range(P_gs):
            chromosome = self.global_selection()
            X1.append(chromosome)

        P_ls = int(self.P * self.LS)
        for idx_chromosome in range(P_ls):
            chromosome = self.local_selection()
            X1.append(chromosome)

        P_rs = int(self.P * self.RS)
        for idx_chromosome in range(P_rs):
            chromosome = self.random_selection()
            X1.append(chromosome)

        X1 = np.array(X1)

        # 初始化 : OS
        spam = self.table_pd['job'].values
        X2 = []
        for i in range(self.P):
            np.random.shuffle(spam)
            X2.append(spam.copy())

        X2 = np.array(X2)

        # 合併 : MS、OS
        self.X = np.hstack([X1, X2])

        # 適應值計算
        F = self.fitness(self.X)

        # 迭代
        for g in range(self.G):
            # 更新最佳解
            if np.min(F) < self.F_gbest:
                idx = F.argmin()
                self.X_gbest = self.X[idx].copy()
                self.F_gbest = F.min()

            # 收斂曲線
            self.loss_curve[g] = self.F_gbest

            # 選擇
            X_new = np.zeros_like(self.X)
            for i in range(self.P):
                X_new[i] = self.selection(self.X, F)
            self.X = X_new.copy()

            # 交配
            for i in range(self.P):
                p = np.random.uniform()
                if p < self.pc:
                    p_idx = np.random.choice(self.P, size=2, replace=False)
                    p1 = self.X[p_idx[0]]
                    p2 = self.X[p_idx[1]]

                    # MS
                    r = np.random.uniform()
                    if r <= self.pTPX:
                        p1[:self.operation], p2[:self.operation] = self.TPX(p1[:self.operation], p2[:self.operation])
                    else:
                        p1[:self.operation], p2[:self.operation] = self.UX(p1[:self.operation], p2[:self.operation])

                    # OS
                    p1[self.operation:], p2[self.operation:] = self.POX(p1[self.operation:], p2[self.operation:])
                self.X[p_idx[0]] = p1
                self.X[p_idx[1]] = p2

            # 突變
            for i in range(self.P):
                p = np.random.uniform()
                if p < self.pm:
                    p1 = self.X[i]
                    p1[:self.operation] = self.machine_mutation(p1[:self.operation])
                    p1[self.operation:] = self.swap_mutation(p1[self.operation:])

            # 適應值計算
            F = self.fitness(self.X)

    # 初始化 1 (global selection, 作者自己發明的)
    def global_selection(self):
        sequence = np.random.choice(self.job, size=self.job, replace=False)
        MS = []
        time_array = np.zeros(self.operation)

        for idx_job in sequence:
            mask = self.table_pd['job'] == idx_job
            table = self.table_pd[mask]

            for idx, row in table.iterrows():
                processing_time = row.iloc[:-2].values
                added_time = time_array + processing_time
                selected_machine = added_time.argmin()
                time_array[selected_machine] = added_time[selected_machine]
                MS.append(selected_machine)

        return MS

    # 初始化 2 (local selection, 作者自己發明的)
    def local_selection(self):
        sequence = np.random.choice(self.job, size=self.job, replace=False)
        MS = []

        for idx_job in sequence:
            time_array = np.zeros(self.operation)
            mask = self.table_pd['job'] == idx_job
            table = self.table_pd[mask]

            for idx, row in table.iterrows():
                processing_time = row.iloc[:-2].values
                added_time = time_array + processing_time
                selected_machine = added_time.argmin()
                time_array[selected_machine] = added_time[selected_machine]
                MS.append(selected_machine)

        return MS

    # 初始化 3 (random selection, 作者自己發明的)
    def random_selection(self):
        sequence = np.random.choice(self.job, size=self.job, replace=False)
        MS = []

        for idx_job in sequence:
            mask = self.table_pd['job'] == idx_job
            table = self.table_pd[mask]

            for idx, row in table.iterrows():
                processing_time = row.iloc[:-2].values
                spam = np.where(processing_time != np.inf)[0]
                selected_machine = np.random.choice(spam, size=1)[0]
                MS.append(selected_machine)

        return MS

    # 選擇 (tournament selection, 競爭式選擇)
    def selection(self, X, F, num=3):
        mask = np.random.choice(self.P, size=num, replace=True)
        F_selected = F[mask]
        X_selected = X[mask]
        c1_idx = F_selected.argmin()
        c1 = X_selected[c1_idx]

        return c1

    # 交配 1 (two-point crossover, 雙點交配)
    def TPX(self, p1, p2):
        # 取得任意兩點
        D = len(p1)
        boundary = np.random.choice(D, size=2, replace=False)
        boundary.sort()
        start, end = boundary[0], boundary[1]

        # 交換
        c1 = p1.copy()
        c2 = p2.copy()
        c1[start:end] = p2[start:end]
        c2[start:end] = p1[start:end]

        return c1, c2

    # 交配 2 (uniform crossover, 均勻交配)
    def UX(self, p1, p2):
        # 隨機選定欲交換的位置
        D = len(p1)
        mask = np.random.choice(2, size=D, replace=True).astype(bool)

        # 交換
        c1 = p1.copy()
        c2 = p2.copy()
        c1[mask] = p2[mask]
        c2[mask] = p1[mask]

        return c1, c2

    # 交配 3 (precedence preserving order-based crossover, POX)
    def POX(self, p1, p2):
        # 提取所有元素的唯一值，並且打亂
        operation_set = np.unique(np.hstack([p1, p2]))
        np.random.shuffle(operation_set)
        # 初始化
        D = len(p1)
        c1 = np.zeros(D) - 1
        c2 = np.zeros(D) - 1

        # 隨機選定欲保留的元素
        spam1 = np.random.choice(2, size=len(operation_set), replace=True).astype(bool)
        Js1 = operation_set[spam1]
        mask1 = np.isin(p1, Js1)
        # 交換
        c1[mask1] = p1[mask1]
        c1[~mask1] = p2[~np.isin(p2, Js1)]

        # 隨機選定欲保留的元素
        spam2 = np.random.choice(2, size=len(operation_set), replace=True).astype(bool)
        Js2 = operation_set[spam2]
        mask2 = np.isin(p2, Js2)
        # 交換
        c2[mask2] = p2[mask2]
        c2[~mask2] = p1[~np.isin(p1, Js2)]

        return c1, c2

    # 突變 1 (作者自己發明的)
    def machine_mutation(self, p1):
        # 為每一位置產生亂數
        D = len(p1)
        c1 = p1.copy()
        r = np.random.uniform(size=D)

        for idx, val in enumerate(p1):
            # 若亂數小於等於突變率，則對該位置進行突變 (放入最小時間的機台)
            if r[idx] <= self.pm:
                alternative_machine_set = self.table_np[idx]
                shortest_machine = alternative_machine_set.argmin()
                c1[idx] = shortest_machine

        return c1

    # 突變 2 (swap mutation, 交換突變)
    def swap_mutation(self, p1):
        # 為每一位置產生亂數
        D = len(p1)
        c1 = p1.copy()
        r = np.random.uniform(size=D)

        for idx1, val in enumerate(p1):
            # 若亂數小於等於突變率，則對該位置進行突變 (隨機與其他位置交換)
            if r[idx1] <= 0.5:
                idx2 = np.random.choice(np.delete(np.arange(D), idx1))
                c1[idx1], c1[idx2] = c1[idx2], c1[idx1]

        return c1

    def plot_curve(self):
        plt.figure()
        plt.title('loss curve ['+str(round(self.gBest_curve[-1], 3))+']')
        plt.plot(self.gBest_curve, label='loss')
        plt.grid()
        plt.legend()
        plt.show()
