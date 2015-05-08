import numpy, scipy.optimize

def maximize(func, x0=None, func_prime=None, bounds=None):
    # Multiply by -1 since we want to maximize
    f = lambda x: -func(x)

    if x0 is None:
        raise ValueError("x0 must be provided")
    if func_prime is None:
        raise ValueError("func_prime must be provided")

    # Ravel func_prime
    fprime = lambda x: -func_prime(x)
    return scipy.optimize.fmin_l_bfgs_b(f, x0, fprime, bounds=bounds)[0]
