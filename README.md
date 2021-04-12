# truncnorm

truncnorm is a library for estimating statistics of and sampling from truncated Normal and Student's t-distributions. Implemented by Eleanor Quint

This package is a port of the Truncated Normal and Student's t-distribution toolbox in Matlab. No guarantees are provided as to their equivalence.

The original toolbox includes:
1. Fast random number generators from the truncated univariate and multivariate student/normal distributions;
2. (Quasi-) Monte Carlo estimator of the cumulative distribution function of the multivariate student/normal;
3. Accurate computation of the quantile function of the normal distribution in the extremes of its tails.

### API

- `mv_normal_cdf(l, u, sigma, n)`: CDF of the truncated multivariate normal cumulative distribution. Computes an estimator of the probability $Pr(l<X<u)$, where $X$ is a zero-mean multivariate normal vector with covariance matrix $\Sig$, that is, $X\~N(0,\Sig)$. For $Y\~N(m,Sig)$, call with `mvNcdf(Sig,l-m,u-m,n)`.

### Citations

- Zdravko Botev (2021). Truncated Normal and Student's t-distribution toolbox (https://www.mathworks.com/matlabcentral/fileexchange/53796-truncated-normal-and-student-s-t-distribution-toolbox), MATLAB Central File Exchange. Retrieved April 3, 2021.
- Z. I. Botev (2017), The Normal Law Under Linear Restrictions: Simulation and Estimation via Minimax Tilting, Journal of the Royal Statistical Society, Series B, Volume 79, Part 1, pp. 1-24
