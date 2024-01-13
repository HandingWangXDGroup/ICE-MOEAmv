'''
常用的填充采样函数： PI、EI、LCB、UCB、SMC

mu: 预测值
sigma：不确定性
fb: 当前最优采样值

'''

import numpy as np
from scipy.stats import norm

def PI(mu, sigma, fb, eps=0):
    Z = (fb - mu - eps)/sigma
    return norm.cdf(Z)


def EI(mu, sigma, fb, eps=0):
    Z = (fb - mu - eps)/sigma
    return (fb - mu - eps)*norm.cdf(Z) + sigma * norm.pdf(Z)


def LCB(mu, sigma, t, d, v=1, delta=.1):
    '''
        :param t: number of iteration
        :param d: dimension of optimization space
        :param v: hyperparameter v = 1*
        :param delta: small constant (prob of regret)
        :return:
        '''
    Kappa = np.sqrt(v * (2 * np.log((t ** (d / 2. + 2)) * (np.pi ** 2) / (3. * delta))))
    return mu - Kappa * sigma


def UCB(mu, sigma, t, d, v=1, delta=.1):
    '''
    :param t: number of iteration
    :param d: dimension of optimization space
    :param v: hyperparameter v = 1*
    :param delta: small constant (prob of regret)
    :return:
    '''
    Kappa = np.sqrt(v * (2 * np.log((t ** (d / 2. + 2)) * (np.pi ** 2) / (3. * delta))))
    return mu + Kappa * sigma


def SMC(mu, sigma):
    # smc1: r = np.random.uniform(-2, 0, size=mu.shape)
    r = np.random.uniform(-2, 0, size=mu.shape[1])
    # smc3: r = np.random.uniform(-2, 0)
    # r = np.random.uniform(-2, 0, size=mu.shape)
    return mu + r*sigma

