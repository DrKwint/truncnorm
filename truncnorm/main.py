"""Port of the Matlab Truncated Normal and Student's t-distribution toolbox v2.0 by Zdravko Botev"""
import numpy as np
from scipy.special import erfc, erfcx, erfcinv
from scipy.optimize import root

DTYPE = np.float64


def ln_phi(x):
    """computes logarithm of tail of Z~N(0,1) mitigating numerical roundoff errors;"""
    return -0.5 * x**2 - np.log(2.) + np.log(erfcx(x / np.sqrt(2)))


def ln_normal_pr(a, b):
    """computes ln(P(a<Z<b)) where Z~N(0,1) very accurately for any 'a', 'b'"""
    p = np.zeros_like(a)

    # case b > a > 0
    idxs_pos = a > 0
    if np.any(idxs_pos):
        pa = ln_phi(a[idxs_pos])  # log of upper tail
        pb = ln_phi(b[idxs_pos])
        p[idxs_pos] = pa + np.log1p(-1 * np.exp(pb - pa))

    # case a < b < 0
    idxs_neg = b < 0
    if np.any(idxs_neg):
        pa = ln_phi(-1 * a[idxs_neg])  # log of upper tail
        pb = ln_phi(-1 * b[idxs_neg])
        p[idxs_neg] = pb + np.log1p(-1 * np.exp(pa - pb))

    # case a < 0 < b
    idxs = np.logical_and(np.logical_not(idxs_pos), np.logical_not(idxs_neg))
    if np.any(idxs):
        pa = erfc(-a[idxs] / np.sqrt(2.)) / 2  # lower tail
        pb = erfc(b[idxs] / np.sqrt(2.)) / 2  # upper tail
        p[idxs] = np.log1p(-pa - pb)
    return p


def cholperm(sigma, l, u):
    """
    Computes permuted lower Cholesky factor L for Sig
    %  by permuting integration limit vectors l and u.
    %  Outputs perm, such that Sig(perm,perm)=L*L'.
    %
    % Reference:
    %  Gibson G. J., Glasbey C. A., Elston D. A. (1994),
    %  "Monte Carlo evaluation of multivariate normal integrals and
    %  sensitivity to variate ordering",
    %  In: Advances in Numerical Methods and Applications, pages 120--126
    """
    d = len(l)
    L = np.zeros((d, d))
    z = np.zeros((d, 1))
    perm = np.arange(d)
    for j in range(d):
        pr = np.full((d, 1), np.inf)
        I = range(j, d)
        D = np.expand_dims(np.diag(sigma), 1)
        s = D[I] - np.sum(L[I, :j]**2, axis=1, keepdims=True)
        s[s < 0] = np.finfo(float).eps
        s = np.sqrt(s)
        tl = (l[I] - np.matmul(L[I, :j], z[:j])) / s
        tu = (u[I] - np.matmul(L[I, :j], z[:j])) / s
        pr[I] = ln_normal_pr(tl, tu)
        k = np.argmin(pr)
        dummy = pr[k]
        jk = [j, k]
        kj = [k, j]
        # update rows and cols of sigma
        sigma[jk, :] = sigma[kj, :]
        sigma[:, jk] = sigma[:, kj]
        # update only rows of L
        L[jk, :] = L[kj, :]
        # update integration limits
        l[jk] = l[kj]
        u[jk] = u[kj]
        perm[jk] = perm[kj]  # keep track of permutation
        # construct L sequentially via Cholesky computation
        s = sigma[j, j] - np.sum(L[j, :j]**2)
        if s < -0.01:
            raise Exception('Sigma is not positive semi-definite')

        if s < 0:
            s = np.finfo(DTYPE).eps
        L[j, j] = np.sqrt(s)
        L[j + 1:d,
          j] = (sigma[j + 1:d, j] - np.matmul(L[j + 1:d, :j],
                                              (L[j, :j]).conj().T)) / L[j, j]
        # find mean value, z(j), of truncated normal:
        tl = (l[j] - np.matmul(L[j, :j], z[:j])) / L[j, j]
        tu = (u[j] - np.matmul(L[j, :j], z[:j])) / L[j, j]
        w = ln_normal_pr(
            tl, tu)  # aids in computing expected value of trunc. normal
        z[j] = (np.exp(-.5 * tl**2 - w) - np.exp(-.5 * tu**2 - w)) / np.sqrt(
            2 * np.pi)
    return L, l, u, perm


def grad_psi(y, L, l, u, compute_jacobian=False):
    # implements gradient of psi(x) to find optimal exponential twisting;
    # assumes scaled 'L' with zero diagonal;
    y = y.astype(np.double)
    L = L.astype(np.double)
    l = l.astype(np.double)
    u = u.astype(np.double)
    d = u.shape[0]
    c = np.zeros([d, 1])
    x = np.zeros([d, 1])
    mu = np.zeros([d, 1])
    x[:d - 1] = np.expand_dims(y[:d - 1], 1)
    mu[:d - 1] = np.expand_dims(y[d - 1:], 1)
    # compute now ~l and ~u
    c[1:d] = np.matmul(L[1:d, :], x)
    lt = l - mu - c
    ut = u - mu - c
    # compute gradients avoiding catastrophic cancellation
    w = ln_normal_pr(lt, ut)
    pl = np.exp(-0.5 * lt**2 - w) / np.sqrt(2 * np.pi)
    pu = np.exp(-0.5 * ut**2 - w) / np.sqrt(2 * np.pi)
    P = pl - pu
    # output the gradient
    dfdx = -mu[:d - 1] + np.matmul(P.conj().T, L[:, :d - 1]).conj().T
    dfdm = mu - x + P
    grad = np.concatenate([dfdx, dfdm[:d - 1]])
    if compute_jacobian:
        lt[np.isinf(lt)] = 0
        ut[np.isinf(ut)] = 0
        dP = -P**2 + lt * pl - ut * pu
        # dPdm
        DL = np.tile(dP, [1, d]) * L
        mx = -np.eye(d) + DL
        xx = np.matmul(L.conj().T, DL)
        mx = mx[:d - 1, :d - 1]
        xx = xx[:d - 1, :d - 1]
        a = np.concatenate([xx, mx.conj().T], axis=1)
        b = np.concatenate([mx, np.eye(d - 1) * (1 + dP[:d - 1])], axis=1)
        J = np.concatenate([a, b], axis=0)
        return np.squeeze(grad), J
    return np.squeeze(grad)


def ntail(l, u):
    """
    % samples a column vector of length=length(l)=length(u)
    % from the standard multivariate normal distribution,
    % truncated over the region [l,u], where l>0 and
    % l and u are column vectors;
    % uses acceptance-rejection from Rayleigh distr.
    % similar to Marsaglia (1964);
    """
    c = l**2 / 2
    n = len(l)
    f = np.expm1(c - u**2 / 2)
    x = c - np.log(1 + np.random.rand(n) * f)  # reallog
    # sample using Rayleigh
    # keep list of rejected
    I = np.arange(x.shape[0])[np.random.rand(n)**2 * x > c]
    d = len(I)
    while d > 0:  # while there are rejections
        cy = c[I]
        # find the thresholds of rejected
        y = cy - np.log(1 + np.random.rand(d) * f[I])  # reallog
        idx = np.arange(y.shape[0])[np.random.rand(d)**2 * y < cy]
        # accepted
        x[I[idx]] = y[idx]
        # store the accepted
        I = I[~idx]
        # remove accepted from list
        d = len(I)
        # number of rejected
    x = np.sqrt(2 * x)
    # this Rayleigh transform can be delayed till the end
    return x


def tn(l, u):
    """
    % samples a column vector of length=length(l)=length(u)
    % from the standard multivariate normal distribution,
    % truncated over the region [l,u], where -a<l<u<a for some
    % 'a' and l and u are column vectors;
    % uses acceptance rejection and inverse-transform method;
    """
    tol = 2
    # controls switch between methods
    # threshold can be tuned for maximum speed for each platform
    # case: abs(u-l)>tol, uses accept-reject from randn
    I = abs(u - l) > tol
    x = l
    if np.any(I):
        tl = l[I]
        tu = u[I]
        x[I] = trnd(tl, tu)
    # case: abs(u-l)<tol, uses inverse-transform
    I = ~I
    if any(I):
        tl = l[I]
        tu = u[I]
        pl = erfc(tl / np.sqrt(2)) / 2
        pu = erfc(tu / np.sqrt(2)) / 2
        x[I] = np.sqrt(2) * erfcinv(2 * (pl -
                                         (pl - pu) * np.random.rand(tl.size)))
    return x


def trnd(l, u):
    # uses acceptance rejection to simulate from truncated normal
    x = np.random.randn(l.size)
    # sample normal
    # keep list of rejected
    I = np.arange(x.shape[0])[np.logical_or(x < l, x > u)]
    d = len(I)
    while d > 0:  # while there are rejections
        ly = l[I]
        # find the thresholds of rejected
        uy = u[I]
        y = np.random.randn(ly.size)
        idx = np.logical_and(y > ly, y < uy)
        # accepted
        x[I[idx]] = y[idx]
        # store the accepted
        I = I[~idx]
        # remove accepted from list
        d = len(I)
        # number of rejected
    return x


def trandn(l, u):
    """
    %% truncated normal generator
    % * efficient generator of a vector of length(l)=length(u)
    % from the standard multivariate normal distribution,
    % truncated over the region [l,u];
    % infinite values for 'u' and 'l' are accepted;
    % * Remark:
    % If you wish to simulate a random variable
    % 'Z' from the non-standard Gaussian N(m,s^2)
    % conditional on l<Z<u, then first simulate
    % X=trandn((l-m)/s,(u-m)/s) and set Z=m+s*X;
    %
    % See also: norminvp
    %
    % For more help, see <a href="matlab:
    % doc">Truncated Multivariate Student & Normal</a> documentation at the bottom.
    % Reference: Z. I. Botev (2017), _The Normal Law Under Linear Restrictions:
    % Simulation and Estimation via Minimax Tilting_, Journal of the Royal
    % Statistical Society, Series B, Volume 79, Part 1, pp. 1-24
    """
    if np.any(l > u):
        raise Exception(
            'Truncation limits have to be vectors of the same length with l<u')
    x = np.zeros_like(l)
    x.fill(np.nan)
    a = .66
    # treshold for switching between methods
    # threshold can be tuned for maximum speed for each Matlab version
    # three cases to consider:
    # case 1: a<l<u
    I = l > a
    if np.any(I):
        tl = l[I]
        tu = u[I]
        x[I] = ntail(tl, tu)
    # case 2: l<u<-a
    J = u < -a
    if any(J):
        tl = -u(J)
        tu = -l(J)
        x[J] = -ntail(tl, tu)
    # case 3: otherwise use inverse transform or accept-reject
    I = np.logical_not(np.logical_or(I, J))
    if np.any(I):
        tl = l[I]
        tu = u[I]
        x[I] = tn(tl, tu)
    return x


def mv_normal_pr(n, L, l, u, mu):
    """
    % computes P(l<X<u), where X is normal with
    % 'Cov(X)=L*L' and zero mean vector;
    % exponential tilting uses parameter 'mu';
    % Monte Carlo uses 'n' samples;
    """
    d = len(l)
    mu = np.append(mu, 0)
    Z = np.zeros([d, n])
    p = 0
    for k in range(d - 1):
        col = np.matmul(L[k, :k], Z[:k, :])
        tl = l[k] - mu[k] - col
        tu = u[k] - mu[k] - col
        #simulate N(mu,1) conditional on [tl,tu]
        Z[k] = mu[k] + trandn(tl, tu)
        # update likelihood ratio
        p = p + ln_normal_pr(tl, tu) + 0.5 * mu[k]**2 - mu[k] * Z[k, :]
    col = np.matmul(L[d - 1, :], Z)
    tl = l[d - 1] - col
    tu = u[d - 1] - col
    p = p + ln_normal_pr(tl, tu)
    p = np.exp(p)
    prob = np.mean(p)
    rel_err = np.std(p) / np.sqrt(n) / prob
    return prob, rel_err


def psy(x, L, l, u, mu):
    # implements psi(x,mu); assumes scaled 'L' without diagonal;
    x = np.append(x, 0)
    mu = np.append(mu, 0)
    # compute now ~l and ~u
    c = np.matmul(L, x)
    l = np.squeeze(l) - mu - c
    u = np.squeeze(u) - mu - c
    p = np.sum(ln_normal_pr(l, u) + 0.5 * mu**2 - x * mu)
    return p


def mv_normal_cdf(l, u, sigma, n):
    """
    %% truncated multivariate normal cumulative distribution
    % computes an estimator of the probability Pr(l<X<u),
    % where 'X' is a zero-mean multivariate normal vector
    % with covariance matrix 'Sig', that is, X~N(0,Sig)
    % infinite values for vectors 'u' and 'l' are accepted;
    % Monte Carlo method uses sample size 'n'; the larger
    % the 'n', the smaller the relative error of the estimator;
    %      output: tuple with
    %              1. estimated value of probability Pr(l<X<u)
    %              2. estimated relative error of estimator
    %              3. theoretical upper bound on true Pr(l<X<u)
    %              Remark: If you want to estimate Pr(l<Y<u),
    %                   where Y~N(m,Sig) has mean vector 'm',
    %                     then use 'mvNcdf(Sig,l-m,u-m,n)'.
    % * Example:
    % ```python
    % d = 25
    % l = np.ones([d, 1]) / 2
    % u = np.ones([d, 1])
    % Sig = (0.5 * np.eye(d) + .5 * np.ones([d, d])).T
    % est = mv_normal_cdf(l, u, Sig, 10 ^ 4)  # output of our method
    % ```
    %
    % See also: mvNqmc, mvrandn
    %
    % Ported from the Matlab "Truncated Multivariate Student & Normal" Toolbox.
    % Reference: Z. I. Botev (2017), _The Normal Law Under Linear Restrictions:
    % Simulation and Estimation via Minimax Tilting_, Journal of the Royal
    % Statistical Society, Series B, Volume 79, Part 1, pp. 1-24
    """
    # Cast non-integral inputs to DTYPE
    l = l.astype(DTYPE)
    u = u.astype(DTYPE)
    sigma = sigma.astype(DTYPE)

    d = len(l)
    L, l, u, _ = cholperm(sigma, l, u)
    D = np.diag(L)
    if np.any(D < np.finfo(np.float).eps):
        raise Exception(
            "mv_normal_cdf may fail as covariance matrix is singular!")
    L = L / np.tile(D, [d, 1]).T
    u = u / np.expand_dims(D, 1)
    l = l / np.expand_dims(D, 1)
    L = L - np.eye(d)
    # find optimal tilting parameter via non-linear equation solver
    soln = root(fun=lambda x: grad_psi(x, L, l, u, True),
                x0=np.zeros(2 * (d - 1), dtype=DTYPE),
                method='lm',
                jac=True,
                tol=1e-6)

    x = soln.x[:d - 1]
    mu = soln.x[d - 1:(2 * d - 1)]  # assign saddlepoint x* and mu*
    est, rel_err = mv_normal_pr(n, L, l, u, mu)  # compute psi star
    # calculate an upper bound
    upbnd = psy(x, L, l, u, mu)
    if upbnd < -743:
        raise Exception(
            'Natural log of upbnd probability is less than -743, yielding 0 after exponentiation!'
        )
    upbnd = np.exp(upbnd)
    return est, rel_err, upbnd
