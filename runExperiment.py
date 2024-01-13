from ICE_MOEAmv import ICEMOEAmv
from MVMOP import *
from multiprocessing.pool import Pool
from multiprocessing import Lock
from tqdm import tqdm
from indicators.metrics import igd_plus

def test(args):
    k, prob = args
    print("Running on {} start!(process {})".format(prob.name, k+1))
    opt = ICEMOEAmv(prob=prob, MaxFEs=300, init_size=100, rs=k + 1)
    X, F = opt.run()
    print(igd_plus(F, prob.pareto_front()))
    print("Running on {} end!(process {})".format(prob.name, k+1))

if __name__ == "__main__":

    # 测试问题
    p_lst = []
    for m_type in ['uniform', 'non-uniform']: # 'non-uniform'
        for p in [ZDT1, ZDT2, ZDT3, ZDT4, ZDT6, DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7]:
            for theta in [ 0.2, 0.5, 0.8]:  # 
                p_lst.append(p(n_var=10, theta=theta, method=m_type))

    M = 1
    for prob in p_lst:
        with Pool(M, initializer=tqdm.set_lock, initargs=(Lock(),)) as pool:  #
            pool.map(test, [(i, prob) for i in range(M)])