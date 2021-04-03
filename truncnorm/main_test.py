import numpy as np

from truncnorm.main import mv_normal_cdf


def author_test():
    d = 25
    l = np.ones([d, 1]) / 2
    u = np.ones([d, 1])
    Sig = (0.5 * np.eye(d) + .5 * np.ones([d, d])).T
    est = mv_normal_cdf(l, u, Sig, 10 ^ 4)  # output of our method
    print(est)
