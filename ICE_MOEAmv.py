import os
import math

from scipy.spatial.distance import cdist
from smt.surrogate_models import KRG
from smt.applications.mixed_integer import MixedIntegerSurrogateModel, ORD, FLOAT, ENUM
from AF.infillsample import EI

from MVMOP import *

import h5py

import warnings
warnings.filterwarnings('ignore')


class Pop(object):
    def __init__(self, X):
        self.X = X
        self.ObjV = None


class ICEMOEAmv(object):
    def __init__(self, prob, MaxFEs, init_size, rs):

        # 问题定义
        self.n_var = prob.n_var
        self.n_d = prob.n_d
        self.n_c = prob.n_c
        self.n_obj = prob.n_obj
        self.prob = prob
        self.func = prob.evaluate
        self.xmin = prob.bounds()[0]
        self.xmax = prob.bounds()[1]
        self.N_lst = self.xmax[:self.n_d].astype(int) + 1  # 离散取值数量

        # 算法参数
        self.name = "ICE-MOEAmv"
        self.MaxFEs = MaxFEs
        self.init_size = init_size

        self.FEs = 0
        self.arch = None
        self.rs = rs

        self.proC = 1
        self.disC = 20
        self.proM = 1
        self.disM = 20

        self.w_max = 50
        self.K2 = 4
        self.kappa = 0.05

        self.sm_lst = [None for _ in range(self.n_obj)]
        self.database = None

    def initialization(self):
        X = np.zeros((self.init_size, self.n_var))
        area = self.xmax - self.xmin

        np.random.seed(self.rs)
        for j in range(self.n_d):
            for i in range(self.init_size):
                X[i, j] = int(np.random.uniform(i / self.init_size * self.N_lst[j],
                                                (i + 1) / self.init_size * self.N_lst[j]))
            np.random.shuffle(X[:, j])

        for j in range(self.n_d, self.n_var):
            for i in range(self.init_size):
                X[i, j] = self.xmin[j] + np.random.uniform(i / self.init_size * area[j],
                                                           (i + 1) / self.init_size * area[j])
            np.random.shuffle(X[:, j])
        np.random.seed()
        self.arch = Pop(X)
        self.arch.ObjV = self.func(X)
        self.FEs = self.init_size
        self.database = [self.arch.X, self.arch.ObjV]

        Next, frontno, cd = self.NSGAII_selection(self.arch.ObjV, self.init_size)
        self.arch.X = self.arch.X[Next]
        self.arch.ObjV = self.arch.ObjV[Next]
        self.nd_idxs = np.where(frontno == 1)[0]

    # 更新代理模型
    def update_surrogate(self, archX, archy):
        # 对每个目标构建代理模型
        xtypes = [ORD] * self.n_d + [FLOAT] * self.n_c
        for i in range(self.n_obj):
            self.sm_lst[i] = MixedIntegerSurrogateModel(
                xtypes=xtypes, xlimits=[*zip(self.xmin, self.xmax)],
                surrogate=KRG(theta0=[1e-2], print_global=False),
            )
            self.sm_lst[i].set_training_values(archX, archy[:, i])
            self.sm_lst[i].train()

    def sm_predict(self, newX):
        temp = newX.copy()
        temp[:, :self.n_d] = np.round(newX[:, :self.n_d])

        predobj = np.zeros((newX.shape[0], self.n_obj))
        predunc = np.zeros((newX.shape[0], self.n_obj))
        for i in range(self.n_obj):
            predobj[:, i] = self.sm_lst[i].predict_values(temp).ravel()
            predunc[:, i] = np.sqrt(self.sm_lst[i].predict_variances(temp)).ravel()
        return predobj, predunc

    def RCGA(self, X):
        matingpool = np.arange(X.shape[0])
        np.random.shuffle(matingpool)
        X = X[matingpool]

        X1 = X[0: math.floor(X.shape[0] / 2), :]
        X2 = X[math.floor(X.shape[0] / 2): math.floor(X.shape[0] / 2) * 2, :]
        N = X1.shape[0]
        D = X1.shape[1]

        beta = np.zeros((N, D))
        mu = np.random.random((N, D))
        beta[mu <= 0.5] = (2 * mu[mu <= 0.5]) ** (1 / (self.disC + 1))
        beta[mu > 0.5] = (2 - 2 * mu[mu > 0.5]) ** (-1 / (self.disC + 1))
        beta = beta * (-1) ** np.random.randint(0, 2, (N, D))
        beta[np.random.random((N, D)) < 0.5] = 1
        beta[np.tile(np.random.random((N, 1)) > self.proC, (1, D))] = 1
        Offspring = np.vstack(
            (
                (X1 + X2) / 2 + beta * (X1 - X2) / 2,
                (X1 + X2) / 2 - beta * (X1 - X2) / 2,
            )
        )
        Lower = np.tile(self.xmin[self.n_var - X.shape[1]:], (2 * N, 1))
        Upper = np.tile(self.xmax[self.n_var - X.shape[1]:], (2 * N, 1))
        Site = np.random.random((2 * N, D)) < self.proM / D
        mu = np.random.random((2 * N, D))
        temp = np.logical_and(Site, mu <= 0.5)
        Offspring = np.minimum(np.maximum(Offspring, Lower), Upper)
        Offspring[temp] = Offspring[temp] + (Upper[temp] - Lower[temp]) * (
                (
                        2 * mu[temp]
                        + (1 - 2 * mu[temp])
                        * (  # noqa
                                1
                                - (Offspring[temp] - Lower[temp]) / (Upper[temp] - Lower[temp])  # noqa
                        )
                        ** (self.disM + 1)
                )
                ** (1 / (self.disM + 1))
                - 1
        )  # noqa
        temp = np.logical_and(Site, mu > 0.5)  # noqa: E510
        Offspring[temp] = Offspring[temp] + (Upper[temp] - Lower[temp]) * (
                1
                - (
                        2 * (1 - mu[temp])
                        + 2
                        * (mu[temp] - 0.5)
                        * (  # noqa
                                1
                                - (Upper[temp] - Offspring[temp]) / (Upper[temp] - Lower[temp])  # noqa
                        )
                        ** (self.disM + 1)
                )
                ** (1 / (self.disM + 1))
        )
        Offspring = np.clip(Offspring, self.xmin[self.n_var - X.shape[1]:], self.xmax[self.n_var - X.shape[1]:])
        return Offspring

    def calc_pt(self):
        Xd = self.arch.X[:,:self.n_d]
        x1_d = Xd[:int(0.45 * self.init_size)] if len(self.nd_idxs) >= int(0.45 * self.init_size) else Xd[self.nd_idxs]

        pt = [np.zeros(nk) for nk in self.N_lst]
        for j in range(self.n_d):
            for k in range(self.N_lst[j]):
                ct = np.sum(Xd[:, j] == k)
                ctb = np.sum(x1_d[:, j] == k)
                if (ct == 0):
                    pt[j][k] = 1 / (self.n_d * (self.N_lst[j] ** (self.n_obj - 1)))
                else:
                    pt[j][k] = ctb / ct + 1 / (self.n_d * (self.N_lst[j] ** (self.n_obj - 1)))
        # 归一化
        for i, p in enumerate(pt):
            pt[i] = p / np.sum(p)
        return pt

    def GAEDA(self):
        Xc =  self.arch.X[:, self.n_d:]
        xc = Xc[:int(0.1 * self.init_size)] if len(self.nd_idxs) < int(0.1 * self.init_size) else Xc[self.nd_idxs]

        oxc = self.RCGA(xc)

        pt = self.calc_pt()
        oxd = np.zeros((oxc.shape[0], self.n_d))
        for k in range(oxc.shape[0]):
            for j in range(self.n_d):
                oxd[k, j] = np.random.choice(np.arange(self.N_lst[j]), 1, p=pt[j])
        ox = np.hstack([oxd, oxc])
        return ox

    # 高效非支配排序
    def eff_nds(self, pop_obj):
        # Use efficient non-dominated sort with sequential search (ENS-SS)
        nsort = pop_obj.shape[0]
        # 目标值重复的个体不参与非支配排序
        objv, index, ind = np.unique(
            pop_obj, return_index=True, return_inverse=True, axis=0
        )
        count, M = objv.shape
        frontno = np.full(count, np.inf)
        maxfront = 0
        # 对全部个体进行非支配排序
        Table, _ = np.histogram(ind, bins=np.arange(0, np.max(ind) + 2))
        while np.sum(Table[frontno < np.inf]) < np.min((nsort, len(ind))):
            maxfront += 1
            for i in range(count):
                if frontno[i] == np.inf:
                    dominate = False
                    for j in range(i - 1, -1, -1):
                        if frontno[j] == maxfront:
                            m = 1
                            while m < M and objv[i][m] >= objv[j][m]:
                                m += 1
                            dominate = m == M
                            if dominate or M == 2:
                                break
                    if not dominate:
                        frontno[i] = maxfront
        frontno = frontno[ind]
        return [frontno, maxfront]

    # 拥挤度距离选择
    def crowded_distance(self, Fi, pop_obj):
        l = len(Fi)
        m = self.n_obj

        Fi = np.array(Fi).astype(int)
        distI = np.zeros(l)  # 对应Fi上的位置
        for i in range(m):
            sortFinds = np.argsort(pop_obj[Fi, i])  # l个个体在fm上的排序
            distI[sortFinds[0]] = np.inf
            distI[sortFinds[-1]] = np.inf
            for j in range(1, l - 1):
                distI[sortFinds[j]] += (pop_obj[Fi[sortFinds[j + 1]], i] - pop_obj[Fi[sortFinds[j - 1]], i]) / (
                        np.max(pop_obj[:, i]) - np.min(pop_obj[:, i]))
        return distI

    def NSGAII_selection(self, pop_obj, size):

        frontno, _ = self.eff_nds(pop_obj)

        # 计算拥挤度距离
        Next = []
        nextfrontNO = np.zeros(size).astype(int)
        nextcd = np.zeros(size)

        i, c = 1, 0
        Fi = np.where(frontno == 1)[0]
        while (c + len(Fi) < size):
            if len(Fi) > 0:
                Next.extend(Fi.tolist())
                nextfrontNO[c:c + len(Fi)] = i
                nextcd[c:c + len(Fi)] = self.crowded_distance(Fi, pop_obj)
                c += len(Fi)
            i += 1
            Fi = np.where(frontno == i)[0]

        # 最后一层根据cd选
        lastcd = self.crowded_distance(Fi, pop_obj)
        slinds = lastcd.argsort()[::-1]
        Next.extend(Fi[slinds[:size - c]].tolist())
        nextfrontNO[c:] = i
        nextcd[c:] = lastcd[slinds[:size - c]]

        return Next, nextfrontNO, nextcd

    def NBI(self, N: int, M: int):
        """
        生成N个M维的均匀分布的权重向量
        :param N: 种群大小
        :param M: 目标维数
        :return: 返回权重向量和种群大小，种群大小可能有改变
        """
        H1 = 1
        while comb(H1 + M, M - 1, exact=True) <= N:
            H1 += 1
        W = (
                np.array(list(combinations(range(1, H1 + M), M - 1)))
                - np.tile(np.arange(M - 1), (comb(H1 + M - 1, M - 1, exact=True), 1))
                - 1
        )
        W = (
                    np.hstack((W, np.zeros((W.shape[0], 1)) + H1))
                    - np.hstack((np.zeros((W.shape[0], 1)), W))
            ) / H1
        if H1 < M:
            H2 = 0
            while (
                    comb(H1 + M - 1, M - 1, exact=True)
                    + comb(H2 + M, M - 1, exact=True)
                    <= N
            ):
                H2 += 1
            if H2 > 0:
                W2 = (
                        np.array(list(combinations(range(1, H2 + M), M - 1)))
                        - np.tile(
                    np.arange(M - 1), (comb(H2 + M - 1, M - 1, exact=True), 1)
                )
                        - 1
                )
                W2 = (
                             np.hstack((W, np.zeros((W2.shape[0], 1))))
                             - np.hstack((np.zeros((W2.shape[0], 1)), W2))
                     ) / H2
                W = np.vstack((W, W2 / 2 + 1 / (2 * M)))
        W = np.maximum(W, 1e-6)
        return W

    # 基于参考向量筛选
    def RV_selection(self, objv):
        n = objv.shape[0] // 2
        w = self.NBI(n, self.n_obj)

        # 归一化
        zmin = np.min(objv, axis=0)
        zmax = np.max(objv, axis=0)
        objv = objv - zmin / (zmax - zmin)

        # Associate each solution to a reference vector
        Angle = np.arccos(1 - cdist(objv, w, "cosine"))
        associate = np.argmin(Angle, axis=1)
        associate_map = {}
        for i, rvi in enumerate(associate):
            associate_map.setdefault(rvi, []).append(i)

        # Select one solution for each reference vector (一层一层选出n个)
        Next = []
        while len(Next) < n:
            for rvi in list(associate_map.keys()):
                idxs = associate_map[rvi]  # 与向量i联系的个体
                ECD = np.linalg.norm(objv[idxs] - np.min(objv, axis=0), axis=1)
                bestind = np.argmin(ECD)
                Next.append(idxs[bestind])
                associate_map[rvi].pop(bestind)
                if len(associate_map[rvi]) == 0:
                    associate_map.pop(rvi)
                if len(Next) == n:
                    break
        return Next

    def SAMOEA(self):
        x, obj = self.arch.X, self.arch.ObjV

        for t in range(self.w_max):
            ox = self.RCGA(x)
            ox[:, :self.n_d] = np.round(ox[:, :self.n_d])
            of, _ = self.sm_predict(ox)

            cx = np.vstack([x, ox])
            cy = np.vstack([obj, of])

            next_idxs = self.RV_selection(cy)
            x = cx[next_idxs]
            obj = cy[next_idxs]
        return x, obj

    # 计算种群的 EIM 值
    def cal_EIM(self, x, mu, std):
        f_mat = self.arch.ObjV[self.nd_idxs]  # k,m
        n = x.shape[0]

        # 计算个体的 EIM 值
        eim = np.zeros(n)
        for i in range(n):
            EIM = EI(mu[i], std[i], f_mat, 0)
            # 欧氏距离聚合(Euclidean)
            eim[i] = np.min(np.sqrt(np.sum(EIM ** 2, axis=1)))

        return eim

    def cal_indicator(self, x):
        n = x.shape[0]

        # 预测值及不确定性
        mu, std = self.sm_predict(x)

        # 计算 ei
        ei = self.cal_EIM(x, mu, std)

        # 计算多样性指标
        di = np.zeros(n)
        for i in range(n):
            di[i] = np.min(np.linalg.norm(mu[i] - self.arch.ObjV[self.nd_idxs], axis=1))

        ei = (ei - np.min(ei)) / (np.max(ei) - np.min(ei))
        di = (di - np.min(di)) / (np.max(di) - np.min(di))

        c = len(self.nd_idxs) / self.init_size
        return (1-c)*ei + c*di

    # 计算I指标作为适应度值
    def CalFitness(self, Objv):
        N = Objv.shape[0]
        # 归一化
        fmin = np.min(Objv, axis=0)
        fmax = np.max(Objv, axis=0)
        objv = (Objv - fmin) / (fmax - fmin)
        I = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                I[i, j] = np.max(objv[i] - objv[j], axis=0)
        C = np.max(np.abs(I), axis=0)
        Fitness = np.sum(-np.exp(-I / C / self.kappa), axis=0) + 1
        return Fitness, I, C

    def ind_based_selection(self, pop_objv):
        # 若非支配解数量小于popsize, 随机采用Ieps+
        F, I, C = self.CalFitness(pop_objv)

        next = list(range(pop_objv.shape[0]))
        while len(next) > self.K2:
            x = np.argmin(F[next])
            F = F + np.exp(-I[next[x]] / C[next[x]] / self.kappa)
            next.remove(next[x])
        return next

    def rm_dup(self, x):
        # x去重
        uni_x, s = np.unique(x, axis=0, return_index=True)

        # uni_x与database去重
        edis = cdist(uni_x, self.database[0], 'euclidean')
        unidxs = np.where(np.sum(edis, axis=1) > 1e-8)[0]

        return s[unidxs]

    def update_DB_arch(self, in_x, in_y):

        # 更新 database
        self.database[0] = np.vstack([self.database[0], in_x])
        self.database[1] = np.vstack([self.database[1], in_y])

        cx = np.vstack([self.arch.X, in_x])
        cy = np.vstack([self.arch.ObjV, in_y])

        Next, frontno, cd = self.NSGAII_selection(cy, self.init_size)
        self.arch.X = cx[Next]
        self.arch.ObjV = cy[Next]
        self.nd_idxs = np.where(frontno == 1)[0]

        filename = "results/" + self.prob.name + '/' + self.prob.name + '_' + str(
            self.prob.theta) + '_' + self.prob.method + '_' + self.name + '_' + str(self.rs) + ".h5"
        if not os.path.exists(filename):
            f = h5py.File(filename, 'w')
        else:
            f = h5py.File(filename, 'a')

        nds_data = np.hstack([self.arch.X[self.nd_idxs], self.arch.ObjV[self.nd_idxs]])
        f.create_dataset(str(self.FEs - self.init_size).zfill(2), data=nds_data)
        f.close()


    def run(self):

        self.initialization()
        while self.FEs < self.MaxFEs:

            '''
            EIM 优化
            '''
            self.update_surrogate(self.database[0], self.database[1])
            x_glo, _ = self.SAMOEA()
            indicators = self.cal_indicator(x_glo)
            best_ind = np.argmax(indicators)

            in_x1 = x_glo[best_ind].reshape(1,-1)
            in_x1[:, :self.n_d] = np.round(in_x1[:, :self.n_d])
            in_y1 = self.func(in_x1)
            self.FEs += len(in_y1)
            # 更新数据
            self.update_DB_arch(in_x1, in_y1)

            '''
            局部优化
            '''
            self.update_surrogate(self.arch.X, self.arch.ObjV)
            x_loc = self.GAEDA()
            y_loc, _ = self.sm_predict(x_loc)

            loc_next = self.ind_based_selection(y_loc)

            in_x2 = x_loc[loc_next]
            in_x2[:, :self.n_d] = np.round(in_x2[:, :self.n_d])

            uindxs = self.rm_dup(in_x2)
            in_x2 = in_x2[uindxs]

            # 真实评估
            in_y2 = self.func(in_x2)
            self.FEs += len(in_y2)

            # 更新数据
            self.update_DB_arch(in_x2, in_y2)

        bestX = self.arch.X[self.nd_idxs]
        bestf = self.arch.ObjV[self.nd_idxs]

        return bestX, bestf



