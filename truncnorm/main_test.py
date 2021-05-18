import numpy as np

from truncnorm.main import mv_normal_cdf


def author_test():
    d = 25
    l = np.ones([d, 1]) / 2
    u = np.ones([d, 1])
    Sig = np.linalg.inv(0.5 * np.eye(d) + .5 * np.ones([d, d]))
    est = mv_normal_cdf(l, u, Sig, 10000)  # output of our method
    print(est)


if __name__ == '__main__':
    author_test()