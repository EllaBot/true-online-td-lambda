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
        self.multipliers = FourierBasis._multipliers(d, n)

    def get_num_basis_functions(self):
        """Gets the number of basis functions
        """
        if hasattr(self, 'num_functions'):
            return self.num_functions

        self.num_functions = (self.n + 1.0) ** self.d
        return self.num_functions

    @staticmethod
    def _scale(value, ranges, index):
        minimum = float(ranges[index, 0])
        maximum = float(ranges[index, 1])
        if minimum == maximum:
            return 0.0

        return (value - minimum) / (maximum - minimum)

    @staticmethod
    def _multipliers(d, n):
        """Generates multipliers for the fourier basis.
        This corresponds to the c vector in the paper
        """
        arrays = [list(range(n + 1)) for _ in itertools.repeat(None, d)]
        return FourierBasis._cartesian(arrays)

    @staticmethod
    def _cartesian(arrays, out=None):
        """Generate a cartesian product of input arrays.
        Parameters
        ----------
        arrays : list of array-like
            1-D arrays to form the cartesian product of.
        out : ndarray
            Array to place the cartesian product in.
        Returns
        -------
        out : ndarray
            2-D array of shape (M, len(arrays)) containing cartesian products
            formed of input arrays.
        Examples
        --------
        >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
        array([[1, 4, 6],
               [1, 4, 7],
               [1, 5, 6],
               [1, 5, 7],
               [2, 4, 6],
               [2, 4, 7],
               [2, 5, 6],
               [2, 5, 7],
               [3, 4, 6],
               [3, 4, 7],
               [3, 5, 6],
               [3, 5, 7]])

        From scikit-learn's extra math
        """
        arrays = [numpy.asarray(x) for x in arrays]
        shape = (len(x) for x in arrays)
        dtype = arrays[0].dtype

        ix = numpy.indices(shape)
        ix = ix.reshape(len(arrays), -1).T

        if out is None:
            out = numpy.empty_like(ix, dtype=dtype)

        for n, arr in enumerate(arrays):
            out[:, n] = arrays[n][ix[:, n]]

        return out

