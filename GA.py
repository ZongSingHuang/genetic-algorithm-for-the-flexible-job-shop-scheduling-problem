# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 16:13:49 2020

@author: e10832
"""
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)


class GA():
    def __init__(self, fitness, D, P, G=100,
                 GS=0.6, LS=0.3, RS=0.1,
                 pc=0.7, two=0.5, uniform=0.5,
                 pm=0.01):
        self.fitness = fitness
        self.D = D
        self.P = P
        self.G = G
        self.GS = GS
        self.LS = LS
        self.RS = RS
        self.pc = pc
        self.two = two
        self.uniform = uniform
        self.pm = pm

        self.X_pbest = np.zeros([self.P, self.D])
        self.F_pbest = np.zeros([self.P]) + np.inf
        self.X_gbest = np.zeros([self.D])
        self.F_gbest = np.inf
        self.loss_curve = np.zeros(self.G)

    def opt(self):
        # 初始化
        self.X = np.random.uniform(low=self.lb, high=self.ub, size=[self.P, self.D])
        self.V = np.zeros([self.P, self.D])
        
        # 迭代
        for g in range(self.G):
            # 適應值計算
            F = self.fitness(self.X)
            
            # 更新最佳解
            mask = F < self.pbest_F
            self.pbest_X[mask] = self.X[mask].copy()
            self.pbest_F[mask] = F[mask].copy()
            
            if np.min(F) < self.gbest_F:
                idx = F.argmin()
                self.gbest_X = self.X[idx].copy()
                self.gbest_F = F.min()
            
            # 收斂曲線
            self.loss_curve[g] = self.gbest_F
            
            # 更新
            r1 = np.random.uniform(size=[self.P, self.D])
            r2 = np.random.uniform(size=[self.P, self.D])
            w = self.w_max - (self.w_max-self.w_min)*(g/self.G)
            
            self.V = w * self.V + self.c1 * (self.pbest_X - self.X) * r1 \
                                + self.c2 * (self.gbest_X - self.X) * r2 # 更新V
            self.V = np.clip(self.V, -self.v_max, self.v_max) # 邊界處理
            
            self.X = self.X + self.V # 更新X
            self.X = np.clip(self.X, self.lb, self.ub) # 邊界處理
    
    # 雙點交配 (two-point crossover)
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

    # 均勻交配 (uniform crossover)
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

    # precedence preserving orderbased crossover (POX)
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

    # 作者自己發明的
    def machine_mutation(self, p1):
        # 為每一位置產生亂數
        c1 = p1.copy()
        r = np.random.uniform(size=self.D / 2)

        for idx, val in enumerate(p1):
            # 若亂數小於等於突變率，則對該位置進行突變 (放入最小時間的機台)
            if r[idx] <= self.pm:
                alternative_machine_set = self.table_np[idx]
                shortest_machine = alternative_machine_set.argmin()
                c1[idx] = shortest_machine

        return c1

    # 交換突變 (swap mutation)
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