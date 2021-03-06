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
        self.ranges = numpy.array(ranges, dtype=numpy.dtype(float))
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

    def compute_features(self, features):
        """Computes the nth order fourier basis for d variables
        """
        return self.compute_scaled_features(self.scale_features(features))

    def compute_scaled_features(self, scaled_features):
        """Computes the nth order fourier basis for d variables
        """
        if len(scaled_features) == 0:
            return numpy.ones(1)
        return numpy.cos(numpy.pi * numpy.dot(self.multipliers, scaled_features))

    def compute_gradient(self, scaled_features):
        """Computes the gradient of the fourier basis
        """
        if len(scaled_features) == 0:
            return numpy.zeros(1)

        # Calculate outer derivative
        outer_deriv = -numpy.sin(numpy.pi * numpy.dot(self.multipliers, scaled_features))

        # Calculate inner derivative
        # ranges[:, 1] - ranges[:, 0] corresponds to upperbound - lowerbound
        inner_deriv = numpy.pi * self.multipliers

        return (inner_deriv.T * outer_deriv).T

    def scale_features(self, features):
        return numpy.array([FourierBasis._scale(features[i], self.ranges, i) for i in xrange(len(features))])

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

