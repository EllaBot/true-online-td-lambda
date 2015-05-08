import numpy, scipy.optimize

def maximize(func, x0=None, func_prime=None, bounds=None):
    # Multiply by -1 since we want to maximize
    f = lambda x: -func(x)

    if bounds is None:
        raise ValueError("Bounds must be provided")

    fprime = None
    if func_prime is not None:
        fprime = lambda x: -func_prime(x)

    def __l_bfgs_b(*args, **kwargs):
        # Delete unecessary argument from brute
        del kwargs['full_output']
        kwargs['fprime'] = fprime
        kwargs['bounds'] = bounds
        if fprime is None:
            kwargs['approx_grad'] = 1

        output = scipy.optimize.fmin_l_bfgs_b(*args, **kwargs)

        return (output[0], output[1], output[2]['warnflag'])

    # Ranges: step size is default
    return scipy.optimize.brute(f, tuple(bounds),
                                   finish=__l_bfgs_b)
