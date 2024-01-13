import numpy as np

from scipy.special import comb
from itertools import combinations

def uniformpoint(N,M):
    H1=1
    while (comb(H1+M-1,M-1)<=N):
        H1=H1+1
    H1=H1-1
    W=np.array(list(combinations(range(H1+M-1),M-1)))-np.tile(np.array(list(range(M-1))),(int(comb(H1+M-1,M-1)),1))
    W=(np.hstack((W,H1+np.zeros((W.shape[0],1))))-np.hstack((np.zeros((W.shape[0],1)),W)))/H1
    if H1<M:
        H2=0
        while(comb(H1+M-1,M-1)+comb(H2+M-1,M-1) <= N):
            H2=H2+1
        H2=H2-1
        if H2>0:
            W2=np.array(list(combinations(range(H2+M-1),M-1)))-np.tile(np.array(list(range(M-1))),(int(comb(H2+M-1,M-1)),1))
            W2=(np.hstack((W2,H2+np.zeros((W2.shape[0],1))))-np.hstack((np.zeros((W2.shape[0],1)),W2)))/H2
            W2=W2/2+1/(2*M)
            W=np.vstack((W,W2))#按列合并
    W[W<1e-6]=1e-6
    N=W.shape[0]
    return W,N

# 判断支配关系：p支配q，返回1
def is_dominated(p, q):
    if (np.any(p > q)):
        return 0
    elif (np.all(p == q)):
        return 0
    else:
        return 1

def get_nds(ObjV):
    size = len(ObjV)
    F1 = []
    # 寻找pareto第一级个体
    for i in range(size):
        n_p = 0
        for j in range(size):
            if (i != j):
                if (is_dominated(ObjV[j], ObjV[i])):
                    n_p += 1
        if (n_p == 0):
            F1.append(i)
    return F1

class ZDT1(object):
    def __init__(self, n_var, theta, method):

        self.n_var = n_var
        self.theta = theta
        self.n_d = round(n_var * self.theta)
        self.n_c = n_var - self.n_d
        self.n_obj = 2
        self.name = "ZDT1"
        self.method = method

        # 取值集合
        if self.method == "uniform":
            self.Q = 0.05 * np.arange(1, 21)
        elif (self.method == "non-uniform"):
            self.Q = 1/(1+np.exp(5-0.5*np.arange(1, 21)))

    def bounds(self):
        lb = np.zeros(self.n_var)
        ub = np.append((20-1)*np.ones(self.n_d), np.ones(self.n_c))
        return [lb, ub]

    def evaluate(self, x):
        tmp = x.copy()
        for i in range(x.shape[0]):
            for j in range(self.n_d):
                tmp[i, j] = self.Q[round(x[i, j])]

        f1 = tmp[:, 0]
        g = 1 + 9.0 / (self.n_var - 1) * np.sum(tmp[:, 1:], axis=1)
        h = (1 - np.power(f1 / g, 0.5))
        f2 = g * h
        return np.column_stack([f1, f2])

    def pareto_front(self):
        f1 = self.Q
        x2m = np.append([self.Q[0]]*(self.n_d-1), [0]*self.n_c)
        g = 1 + 9.0 / (self.n_var - 1) * np.sum(x2m)
        f2 = g * (1 - np.sqrt(f1/g))
        return np.array([f1, f2]).T

class ZDT2(object):
    def __init__(self, n_var, theta, method):
        self.n_var = n_var
        self.theta = theta
        self.n_d = round(n_var * self.theta)
        self.n_c = n_var - self.n_d
        self.n_obj = 2
        self.name = "ZDT2"
        self.method = method
        # 取值集合
        if self.method == "uniform":
            self.Q = 0.05 * np.arange(1, 21)
        elif (self.method == "non-uniform"):
            self.Q = 1/(1+np.exp(5-0.5*np.arange(1, 21)))

    def bounds(self):
        lb = np.zeros(self.n_var)
        ub = np.append((20-1)*np.ones(self.n_d), np.ones(self.n_c))
        return [lb, ub]

    def evaluate(self, x):
        tmp = x.copy()
        for i in range(x.shape[0]):
            for j in range(self.n_d):
                tmp[i, j] = self.Q[round(x[i, j])]

        f1 = tmp[:, 0]
        g = 1 + 9.0 / (self.n_var - 1) * np.sum(tmp[:, 1:], axis=1)
        h = (1 - np.power(f1 / g, 2))
        f2 = g * h
        return np.column_stack([f1, f2])

    def pareto_front(self):
        f1 = self.Q
        x2m = np.append([self.Q[0]] * (self.n_d - 1), [0] * self.n_c)
        g = 1 + 9.0 / (self.n_var - 1) * np.sum(x2m)
        f2 = g * (1 - np.power(f1/g, 2))
        return np.array([f1, f2]).T


class ZDT3(object):
    def __init__(self, n_var, theta, method):

        self.n_var = n_var
        self.theta = theta
        self.n_d = round(n_var * self.theta)
        self.n_c = n_var - self.n_d
        self.n_obj = 2
        self.name = "ZDT3"
        self.method = method

        # 取值集合
        if self.method == "uniform":
            self.Q = 0.05 * np.arange(1, 21)
        elif (self.method == "non-uniform"):
            self.Q = 1/(1+np.exp(5-0.5*np.arange(1, 21)))
    def bounds(self):
        lb = np.zeros(self.n_var)
        ub = np.append((20 - 1) * np.ones(self.n_d), np.ones(self.n_c))
        return [lb, ub]

    def evaluate(self, x):
        tmp = x.copy()
        for i in range(x.shape[0]):
            for j in range(self.n_d):
                tmp[i, j] = self.Q[round(x[i, j])]
        f1 = tmp[:, 0]
        g = 1 + 9 / (self.n_var - 1) * np.sum(tmp[:, 1:], axis=1)
        h = 1 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10 * np.pi * f1)
        f2 = g * h
        return np.column_stack([f1, f2])

    def pareto_front(self):
        f1 = self.Q
        x2m = np.append([self.Q[0]] * (self.n_d - 1), [0] * self.n_c)
        g = 1 + 9.0 / (self.n_var - 1) * np.sum(x2m)
        f2 = g * (1 - np.sqrt(f1/g) - (f1/g) * np.sin(10 * np.pi * f1))
        PF = np.array([f1, f2]).T
        indxs = get_nds(PF)
        return PF[indxs]

class ZDT4(object):
    def __init__(self, n_var, theta, method):

        self.n_var = n_var
        self.theta = theta
        self.n_d = round(n_var * self.theta)
        self.n_c = n_var - self.n_d
        self.n_obj = 2
        self.name = "ZDT4"
        self.method = method

        # 取值集合
        if self.method == "uniform":
            self.Q = 0.05 * np.arange(1, 21)
        elif (self.method == "non-uniform"):
            self.Q = 1/(1+np.exp(5-0.5*np.arange(1, 21)))

    def bounds(self):
        lb = np.append(np.zeros(self.n_d), -5 * np.ones(self.n_c))
        ub = np.append((20-1) * np.ones(self.n_d), 5 * np.ones(self.n_c))
        return [lb, ub]

    def evaluate(self, x):
        tmp = x.copy()
        for i in range(x.shape[0]):
            for j in range(self.n_d):
                if(j==0):
                    tmp[i, j] = self.Q[round(x[i, j])]
                else:
                    tmp[i, j] = -5 + self.Q[round(x[i, j])] * 10

        f1 = tmp[:, 0]
        g = 1 + 10 * (self.n_var - 1) + np.sum(tmp[:, 1:] ** 2 - 10 * np.cos(4 * np.pi * tmp[:, 1:]), axis=1)
        h = 1 - np.sqrt(f1 / g)
        f2 = g * h
        return np.column_stack([f1, f2])

    def pareto_front(self):
        f1 = self.Q
        x2m = np.append([0] * (self.n_d - 1), [0] * self.n_c)
        g = 1 + 10 * (self.n_var - 1) + np.sum(x2m**2 - 10 * np.cos(4 * np.pi * x2m))
        f2 = g * (1 - np.sqrt(f1 / g))
        return np.array([f1, f2]).T


class ZDT6(object):
    def __init__(self, n_var, theta, method):

        self.n_var = n_var
        self.theta = theta
        self.n_d = round(n_var * self.theta)
        self.n_c = n_var - self.n_d
        self.n_obj = 2
        self.name = "ZDT6"
        self.method = method

        # 取值集合
        if self.method == "uniform":
            self.Q = 0.05 * np.arange(1, 21)
        elif (self.method == "non-uniform"):
            self.Q = 1/(1+np.exp(5-0.5*np.arange(1, 21)))

    def bounds(self):
        lb = np.zeros(self.n_var)
        ub = np.append((20 - 1) * np.ones(self.n_d), np.ones(self.n_c))
        return [lb, ub]

    def evaluate(self, x):
        tmp = x.copy()
        for i in range(x.shape[0]):
            for j in range(self.n_d):
                tmp[i, j] = self.Q[round(x[i, j])]

        f1 = 1 - np.exp(-4 * tmp[:, 0]) * np.power(np.sin(6 * np.pi * tmp[:, 0]), 6)
        g = 1 + 9 * np.power(np.sum(tmp[:, 1:], axis=1) / (self.n_var - 1), 0.25)
        h = (1 - np.power(f1 / g, 2))
        f2 = g * h
        return np.column_stack([f1, f2])

    def pareto_front(self):
        f1 = 1 - np.exp(-4 * self.Q) * np.power(np.sin(6 * np.pi * self.Q), 6)
        x2m = np.append([self.Q[0]] * (self.n_d - 1), [0] * self.n_c)
        g = 1 + 9 * np.power(np.sum(x2m)/(self.n_var-1), 0.25)
        f2 = g * (1 - np.power(f1/g, 2))
        return np.array([f1, f2]).T


class DTLZ1(object):
    def __init__(self, n_var, theta, method):

        # 连续、离散变量数
        self.n_var = n_var
        self.theta = theta
        self.n_d = round(n_var * self.theta)
        self.n_c = n_var - self.n_d
        self.name = "DTLZ1"
        self.method = method

        # 取值集合
        if self.method == "uniform":
            self.Q = 0.05 * np.arange(1, 21)
        elif (self.method == "non-uniform"):
            self.Q = 1/(1+np.exp(5-0.5*np.arange(1, 21)))

        self.n_obj = 3
        self.k = n_var - self.n_obj + 1

    def bounds(self):
        lb = np.zeros(self.n_var)
        ub = np.append((20-1)*np.ones(self.n_d), np.ones(self.n_c))
        return [lb, ub]

    def gm(self, Xm):
        return 100 * (self.k + np.sum((Xm - 0.5) ** 2 - np.cos(20 * np.pi * (Xm - 0.5)) , axis=1))

    def obj_func(self, X, g):
        fs = []
        for i in range(self.n_obj):
            f = 0.5 * (1 + g)
            f *= np.prod(X[:, :X.shape[1] - i], axis=1)
            if i > 0:
                f *= (1 - X[:, X.shape[1] - i])
            fs.append(f)
        return np.column_stack(fs)

    def evaluate(self, x):
        tmp = x.copy()
        for i in range(x.shape[0]):
            for j in range(self.n_d):
                tmp[i, j] = self.Q[round(x[i, j])]

        X, Xm = tmp[:, :self.n_obj - 1], tmp[:, self.n_obj - 1:]
        gm = self.gm(Xm)
        return self.obj_func(X, gm)

    def pareto_front(self):
        k = len(self.Q)
        PF = np.zeros((k**2, self.n_obj))
        for i in range(k):
            for j in range(k):
                PF[i*k + j, :] = [0.5*self.Q[i]*self.Q[j], 0.5*self.Q[i]*(1-self.Q[j]), 0.5*(1-self.Q[i])]
        return PF

class DTLZ2(object):
    def __init__(self, n_var, theta, method):

        # 连续、离散变量数
        self.n_var = n_var
        self.theta = theta
        self.n_d = round(n_var * self.theta)
        self.n_c = n_var - self.n_d
        self.name = "DTLZ2"
        self.method = method

        # 取值集合
        if self.method == "uniform":
            self.Q = 0.05 * np.arange(1, 21)
        elif (self.method == "non-uniform"):
            self.Q = 1/(1+np.exp(5-0.5*np.arange(1, 21)))

        self.n_obj = 3
        self.k = n_var - self.n_obj + 1

    def bounds(self):
        lb = np.zeros(self.n_var)
        ub = np.append((20-1)*np.ones(self.n_d), np.ones(self.n_c))
        return [lb, ub]

    def gm(self, Xm):
        return np.sum((Xm - 0.5) ** 2, axis=1)

    def obj_func(self, X, g):
        fs = []

        for i in range(self.n_obj):
            f = 1 + g
            f *= np.prod(np.cos(X[:, :X.shape[1] - i] * np.pi / 2), axis=1)
            if i > 0:
                f *= np.sin(X[:, X.shape[1] - i] * np.pi / 2)
            fs.append(f)
        return np.column_stack(fs)

    def evaluate(self, x):
        tmp = x.copy()
        for i in range(x.shape[0]):
            for j in range(self.n_d):
                tmp[i, j] = self.Q[round(x[i, j])]

        X, Xm = tmp[:, :self.n_obj - 1], tmp[:, self.n_obj - 1:]
        gm = self.gm(Xm)
        return self.obj_func(X, gm)

    def pareto_front(self):
        k = len(self.Q)
        PF = np.zeros((k ** 2, self.n_obj))
        for i in range(k):
            for j in range(k):
                PF[i * k + j, :] = [np.cos(self.Q[i] * np.pi/2)*np.cos(self.Q[j]* np.pi/2),
                                    np.cos(self.Q[i] * np.pi/2)*np.sin(self.Q[j]* np.pi/2),
                                    np.sin(self.Q[i])]
        return PF

class DTLZ3(object):
    def __init__(self, n_var, theta, method):

        # 连续、离散变量数
        self.n_var = n_var
        self.theta = theta
        self.n_d = round(n_var * self.theta)
        self.n_c = n_var - self.n_d
        self.name = "DTLZ3"
        self.method = method

        # 取值集合
        if self.method == "uniform":
            self.Q = 0.05 * np.arange(1, 21)
        elif (self.method == "non-uniform"):
            self.Q = 1/(1+np.exp(5-0.5*np.arange(1, 21)))

        self.n_obj = 3
        self.k = n_var - self.n_obj + 1

    def bounds(self):
        lb = np.zeros(self.n_var)
        ub = np.append((20 - 1) * np.ones(self.n_d), np.ones(self.n_c))
        return [lb, ub]

    def gm(self, Xm):
        return 100 * (self.k + np.sum((Xm - 0.5) ** 2 - np.cos(20 * np.pi * (Xm - 0.5)), axis=1))

    def obj_func(self, X, g):
        fs = []

        for i in range(self.n_obj):
            f = 1 + g
            f *= np.prod(np.cos(X[:, :X.shape[1] - i] * np.pi / 2), axis=1)
            if i > 0:
                f *= np.sin(X[:, X.shape[1] - i] * np.pi / 2)
            fs.append(f)
        return np.column_stack(fs)

    def evaluate(self, x):
        tmp = x.copy()
        for i in range(x.shape[0]):
            for j in range(self.n_d):
                tmp[i, j] = self.Q[round(x[i, j])]

        X, Xm = tmp[:, :self.n_obj - 1], tmp[:, self.n_obj - 1:]
        gm = self.gm(Xm)
        return self.obj_func(X, gm)

    def pareto_front(self):
        k = len(self.Q)
        PF = np.zeros((k ** 2, self.n_obj))
        for i in range(k):
            for j in range(k):
                PF[i * k + j, :] = [np.cos(self.Q[i] * np.pi / 2) * np.cos(self.Q[j] * np.pi / 2),
                                    np.cos(self.Q[i] * np.pi / 2) * np.sin(self.Q[j] * np.pi / 2),
                                    np.sin(self.Q[i])]
        return PF

class DTLZ4(object):
    def __init__(self, n_var, theta, method):

        # 连续、离散变量数
        self.n_var = n_var
        self.theta = theta
        self.n_d = round(n_var * self.theta)
        self.n_c = n_var - self.n_d
        self.name = "DTLZ4"
        self.method = method

        # 取值集合
        if self.method == "uniform":
            self.Q = 0.05 * np.arange(1, 21)
        elif (self.method == "non-uniform"):
            self.Q = 1/(1+np.exp(5-0.5*np.arange(1, 21)))

        self.n_obj = 3
        self.k = n_var - self.n_obj + 1

    def bounds(self):
        lb = np.zeros(self.n_var)
        ub = np.append((20 - 1) * np.ones(self.n_d), np.ones(self.n_c))
        return [lb, ub]

    def gm(self, Xm):
        return np.sum((Xm - 0.5) ** 2, axis=1)

    def obj_func(self, X, g):
        fs = []

        for i in range(self.n_obj):
            f = 1 + g
            f *= np.prod(np.cos(np.power(X[:, :X.shape[1] - i], 100) * np.pi / 2), axis=1)
            if i > 0:
                f *= np.sin(np.power(X[:, X.shape[1] - i],100) * np.pi / 2)
            fs.append(f)
        return np.column_stack(fs)

    def evaluate(self, x):
        tmp = x.copy()
        for i in range(x.shape[0]):
            for j in range(self.n_d):
                tmp[i, j] = self.Q[round(x[i, j])]

        X, Xm = tmp[:, :self.n_obj - 1], tmp[:, self.n_obj - 1:]
        gm = self.gm(Xm)
        return self.obj_func(X, gm)

    def pareto_front(self):
        k = len(self.Q)
        PF = np.zeros((k ** 2, self.n_obj))
        for i in range(k):
            for j in range(k):
                PF[i * k + j, :] = [np.cos(self.Q[i] * np.pi / 2) * np.cos(self.Q[j] * np.pi / 2),
                                    np.cos(self.Q[i] * np.pi / 2) * np.sin(self.Q[j] * np.pi / 2),
                                    np.sin(self.Q[i])]
        return PF

class DTLZ5(object):
    def __init__(self, n_var, theta, method):

        # 连续、离散变量数
        self.n_var = n_var
        self.theta = theta
        self.n_d = round(n_var * self.theta)
        self.n_c = n_var - self.n_d
        self.name = "DTLZ5"
        self.method = method

        # 取值集合
        if self.method == "uniform":
            self.Q = 0.05 * np.arange(1, 21)
        elif (self.method == "non-uniform"):
            self.Q = 1/(1+np.exp(5-0.5*np.arange(1, 21)))

        self.n_obj = 3
        self.k = n_var - self.n_obj + 1

    def bounds(self):
        lb = np.zeros(self.n_var)
        ub = np.append((20 - 1) * np.ones(self.n_d), np.ones(self.n_c))
        return [lb, ub]

    def gm(self, Xm):
        return np.sum((Xm - 0.5) ** 2, axis=1)

    def obj_func(self, X, g):
        fs = []

        for i in range(self.n_obj):
            f = 1 + g
            f *= np.prod(np.cos(X[:, :X.shape[1] - i] * np.pi / 2), axis=1)
            if i > 0:
                f *= np.sin(X[:, X.shape[1] - i] * np.pi / 2)
            fs.append(f)
        return np.column_stack(fs)

    def evaluate(self, x):
        tmp = x.copy()
        for i in range(x.shape[0]):
            for j in range(self.n_d):
                tmp[i, j] = self.Q[round(x[i, j])]

        X, Xm = tmp[:, :self.n_obj - 1], tmp[:, self.n_obj - 1:]
        gm = self.gm(Xm)
        theta = 1 / (2 * (1 + gm[:, None])) * (1 + 2 * gm[:, None] * X)
        theta = np.column_stack([X[:, 0], theta[:, 1:]])

        return self.obj_func(theta, gm)

    def pareto_front(self):
        k = len(self.Q)
        PF = np.zeros((k, self.n_obj))
        for i in range(k):
            PF[i, :] = [np.cos(self.Q[i] * np.pi / 2) * np.cos(0.5 * np.pi / 2),
                                np.cos(self.Q[i] * np.pi / 2) * np.sin(0.5 * np.pi / 2),
                                np.sin(self.Q[i])]
        return PF


class DTLZ6(object):
    def __init__(self, n_var, theta, method):

        # 连续、离散变量数
        self.n_var = n_var
        self.theta = theta
        self.n_d = round(n_var * self.theta)
        self.n_c = n_var - self.n_d
        self.name = "DTLZ6"
        self.method = method

        # 取值集合
        if self.method == "uniform":
            self.Q = 0.05 * np.arange(1, 21)
        elif (self.method == "non-uniform"):
            self.Q = 1/(1+np.exp(5-0.5*np.arange(1, 21)))

        self.n_obj = 3
        self.k = n_var - self.n_obj + 1

    def bounds(self):
        lb = np.zeros(self.n_var)
        ub = np.append((20 - 1) * np.ones(self.n_d), np.ones(self.n_c))
        return [lb, ub]

    def gm(self, Xm):
        return np.sum(np.power(Xm, 0.1), axis=1)

    def obj_func(self, X, g):
        fs = []

        for i in range(self.n_obj):
            f = 1 + g
            f *= np.prod(np.cos(X[:, :X.shape[1] - i] * np.pi / 2), axis=1)
            if i > 0:
                f *= np.sin(X[:, X.shape[1] - i] * np.pi / 2)
            fs.append(f)
        return np.column_stack(fs)

    def evaluate(self, x):
        tmp = x.copy()
        for i in range(x.shape[0]):
            for j in range(self.n_d):
                tmp[i, j] = self.Q[round(x[i, j])]

        X, Xm = tmp[:, :self.n_obj - 1], tmp[:, self.n_obj - 1:]
        gm = self.gm(Xm)
        theta = 1 / (2 * (1 + gm[:, None])) * (1 + 2 * gm[:, None] * X)
        theta = np.column_stack([X[:,0], theta[:, 1:]])

        return self.obj_func(theta, gm)

    def pareto_front(self):
        k = len(self.Q)
        PF = np.zeros((k, self.n_obj))
        for i in range(k):
            PF[i, :] = [np.cos(self.Q[i] * np.pi / 2) * np.cos(0.5 * np.pi / 2),
                        np.cos(self.Q[i] * np.pi / 2) * np.sin(0.5 * np.pi / 2),
                        np.sin(self.Q[i])]
        return PF

class DTLZ7(object):
    def __init__(self, n_var, theta, method):

        # 连续、离散变量数
        self.n_var = n_var
        self.theta = theta
        self.n_d = round(n_var * self.theta)
        self.n_c = n_var - self.n_d
        self.name = "DTLZ7"
        self.method = method

        # 取值集合
        if self.method == "uniform":
            self.Q = 0.05 * np.arange(1, 21)
        elif (self.method == "non-uniform"):
            self.Q = 1/(1+np.exp(5-0.5*np.arange(1, 21)))

        self.n_obj = 3
        self.k = n_var - self.n_obj + 1

    def bounds(self):
        lb = np.zeros(self.n_var)
        ub = np.append((20 - 1) * np.ones(self.n_d), np.ones(self.n_c))
        return [lb, ub]

    def evaluate(self, x):
        tmp = x.copy()
        for i in range(x.shape[0]):
            for j in range(self.n_d):
                tmp[i, j] = self.Q[round(x[i, j])]

        fs = []
        for i in range(self.n_obj - 1):
            fs.append(tmp[:, i])
        f = np.column_stack(fs)

        g = 1 + 9/self.k * np.sum(tmp[:, -self.k:], axis = 1)
        h = self.n_obj - np.sum(f/(1+g[:,None]) * (1 + np.sin(3*np.pi*f)), axis = 1)

        return np.column_stack([f, (1+g)*h])

    def pareto_front(self):
        interval = np.array([0, 0.251412, 0.631627, 0.859401])
        median = (interval[1] - interval[0]) / (interval[3] - interval[2] + interval[1] - interval[0])

        xlst = np.meshgrid(*[self.Q]*(self.n_obj-1))
        X = np.column_stack([x.ravel() for x in xlst])

        X[X<=median] = X[X<=median]*(interval[1] - interval[0])/median + interval[0]
        X[X>median] = (X[X>median] - median)*(interval[3] - interval[2])/(1-median) + interval[2]

        R = np.column_stack([X, 2*(self.n_obj-np.sum(X/2*(1+np.sin(3*np.pi*X)), axis=1))])
        return R

if __name__ == "__main__":
    prob = ZDT4(n_var=10, theta=0.2, method="uniform")
    pf = prob.pareto_front()
    print(pf)
