import numpy, scipy.optimize

def maximize(func, x0, func_prime, bounds=None):
    # Multiply by -1 since we want to maximize
    f = lambda x: -func(x)
    # Ravel func_prime
    fprime = lambda x: -func_prime(x).ravel()
    return scipy.optimize.fmin_l_bfgs_b(f, x0, fprime, bounds=bounds)[0]
