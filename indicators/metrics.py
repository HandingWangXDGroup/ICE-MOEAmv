import numpy as np
from scipy.spatial.distance import cdist

# from pymoo.indicators.gd import GD
# from pymoo.indicators.igd_plus import IGDPlus
# from pymoo.indicators.igd import IGD
# from pymoo.indicators.hv import HV

def gd(res, opt):
    distance = np.min(cdist(res, opt, metric="euclidean"), axis = 1) # cdist: z -> a
    score = np.linalg.norm(distance)/len(distance)
    return score

def igd(res, opt):
    distance = np.min(cdist(opt, res), axis = 1)
    score = np.mean(distance)
    return score

def igd_plus(res, opt):
    effect = lambda u, v: np.sqrt(np.sum(np.maximum(v-u, 0)**2))  # a-z
    distance = np.min(cdist(opt, res, metric=effect), axis=1)
    score = np.mean(distance)
    return score

def hypervolume(res, opt):
    N, M = res.shape
    fmin = np.minimum(np.min(res, axis=0), 0)
    fmax = np.max(opt, axis = 0)

    # 归一化
    objv = (res-fmin)/((fmax-fmin) * 1.1)
    objv = objv[np.max(objv, axis=1) <= 1]

    RefPoint = np.ones(M)

    if objv.size == 0:
        score = 0

    elif(M < 4):
        # Calculate the exact HV value
        pl =  np.unique(objv, axis=0)
        s = [[1, pl]]

        for k in range(M-1):
            s_ = []
            for i in range(len(s)):
                stemp = Slice(s[i][1], k, RefPoint)
                for j in range(len(stemp)):
                    temp = [[stemp[j][0] * s[i][0], np.array(stemp[j][1])]]
                    s_ = Add(temp, s_)
            s = s_
        score = 0
        for i in range(len(s)):
            p = Head(s[i][1])
            score = score + s[i][0] * np.abs(p[-1] - RefPoint[-1])

    else:
        # Estimate the HV value by Monte Carlo estimation
        sample_num = 1000000
        max_value = RefPoint
        min_value = np.min(objv, axis=0)
        samples = np.random.uniform(
            np.tile(min_value, (sample_num, 1)),
            np.tile(max_value, (sample_num, 1)),
        )
        for i in range(len(objv)):
            domi = np.ones(len(samples), dtype=bool)
            m = 0
            while m <= M - 1 and np.any(domi):
                b = objv[i][m] <= samples[:, m]
                domi = domi & b
                m += 1
            samples = samples[~domi]
        score = np.prod(max_value - min_value) * (
                1 - len(samples) / sample_num
        )

    return score

def Slice(pl: np.ndarray, k: int, ref_point: np.ndarray) -> list:

    p = Head(pl)
    pl = Tail(pl)
    ql = np.array([])
    s = []
    while len(pl):
        ql = Insert(p, k + 1, ql)
        p_ = Head(pl)
        if ql.ndim == 1:
            list_ = [[np.abs(p[k] - p_[k]), np.array([ql])]]
        else:
            list_ = [[np.abs(p[k] - p_[k]), ql]]
        s = Add(list_, s)
        p = p_
        pl = Tail(pl)
    ql = Insert(p, k + 1, ql)
    if ql.ndim == 1:
        list_ = [[np.abs(p[k] - ref_point[k]), [ql]]]
    else:
        list_ = [[np.abs(p[k] - ref_point[k]), ql]]
    s = Add(list_, s)
    return s


def Insert(p: np.ndarray, k: int, pl: np.ndarray) -> np.ndarray:

    flag1 = 0
    flag2 = 0
    ql = np.array([])
    hp = Head(pl)
    while len(pl) and hp[k] < p[k]:
        if len(ql) == 0:
            ql = hp
        else:
            ql = np.vstack((ql, hp))
        pl = Tail(pl)
        hp = Head(pl)
    if len(ql) == 0:
        ql = p
    else:
        ql = np.vstack((ql, p))
    m = max(p.shape)
    while len(pl):
        q = Head(pl)
        for i in range(k, m):
            if p[i] < q[i]:
                flag1 = 1
            elif p[i] > q[i]:
                flag2 = 1
        if not (flag1 == 1 and flag2 == 0):
            if len(ql) == 0:
                ql = Head(pl)
            else:
                ql = np.vstack((ql, Head(pl)))
        pl = Tail(pl)
    return ql


def Head(pl: np.ndarray) -> np.ndarray:
    # 取第一行所有元素
    if pl.ndim == 1:
        p = pl
    else:
        p = pl[0]
    return p


def Tail(pl: np.ndarray) -> np.ndarray:
    # 取除去第一行的所有元素
    if pl.ndim == 1 or min(pl.shape) == 1:
        ql = np.array([])
    else:
        ql = pl[1:]
    return ql


def Add(list_: list, s: list) -> list:

    n = len(s)
    m = 0
    for k in range(n):
        if np.all(list_[0][1]) == np.all(s[k][1]) and len(list_[0][1]) == len(
            s[k][1]
        ):
            s[k][0] = s[k][0] + list_[0][0]
            m = 1
            break
    if m == 0:
        if n == 0:
            s = list_
        else:
            s.append(list_[0])
    s_ = s
    return s_



if __name__ == "__main__":
    np.random.seed(1)
    res = np.random.random((5,2))
    opt = np.random.random((5,2))

    # print([res])
    # print([opt])
    # gd2 = GD(opt)
    # igd2 = IGD(opt)
    # igd_plus2 = IGDPlus(opt)
    # hv2 = HV(ref_point=np.array([1,1]))

    # print(gd(res, opt))
    # print(gd2(res))
    # print(hv2(res))
    print(hypervolume(res, opt))



