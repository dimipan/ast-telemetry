import math

from scipy.stats import binomtest


def sign_test(n_pos, n_neg):
    n = n_pos + n_neg
    if n == 0:
        return float("nan")
    extreme = abs(n_neg - n / 2)
    return min(1.0, sum(math.comb(n, k) * 0.5 ** n for k in range(n + 1) if abs(k - n / 2) >= extreme))


def binomial_greater(hits, n, p):
    return float(binomtest(hits, n, p=p, alternative="greater").pvalue)


def argmax(d):
    return max(d.items(), key=lambda kv: kv[1])[0]
