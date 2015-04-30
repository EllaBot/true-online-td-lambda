import numpy, itertools

class FourierBasis(object):
    """Fourier Basis function approximator [1].

    Parameters
    ----------
    ranges: list
        The ranges of the features

    d: int, optional
        The degree of the fourier basis

    n: int
        The order of the fourier basis

    [1] Konidaris, George. "Value function approximation in reinforcement learning using the Fourier basis." (2008).
    """

    def __init__(self, ranges, d, n = 3):
        self.ranges = numpy.array(ranges)
        self.d = d
        self.n = n

    def get_num_basis_functions(self):
        """Gets the number of basis functions
        """
        if hasattr(self, 'num_functions'):
            return self.num_functions

        self.num_functions = (self.n + 1.0) ** self.d
        return self.num_functions
