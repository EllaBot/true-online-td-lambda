from true_online_td_lambda.basis import FourierBasis

from nose.tools import assert_equal
from nose.tools import assert_list_equal

import numpy as np

class TestFourierBasis(object):

    def test_get_num_basis_functions(self):
        ranges = None
        d = 2
        n = 3
        fourierbasis = FourierBasis(ranges, d, n)
        assert_equal(fourierbasis.get_num_basis_functions(), 16.0)

    def test_cartesian(self):
        arrays = ([0, 1, 2], [0, 1, 2])
        result = FourierBasis._cartesian(arrays).tolist()
        assert_list_equal(result, [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]])

    def test_multipliers(self):
        d = 2
        n = 2
        multipliers = FourierBasis._multipliers(d, n).tolist()
        assert_list_equal(multipliers, [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]])

    def test_scale(self):
        value = 3
        ranges = np.asarray([(0, 10)])
        index = 0
        assert_equal(FourierBasis._scale(value, ranges, index), 0.3)

    def test_compute_features(self):
        ranges = [(0, 1)]
        features = [0.5]
        d = 1
        n = 2
        fourierbasis = FourierBasis(ranges, d, n)
        assert_equal(fourierbasis.compute_features(features)[0], 1.0)

    def test_compute_gradient(self):
        ranges = [(0, 1), (0, 1)]
        features = [0.3, 0.6]
        features_delta_1 = [0.30000001, 0.6] # w.r.t feature 1
        features_delta_2 = [0.3, 0.60000001] # w.r.t feature 2
        d = 2
        n = 2
        fourierbasis = FourierBasis(ranges, d, n)
        # Approximate gradient by derivative approximation
        # (f(x + e) - f(x)) / e
        approxgrad_1 = (fourierbasis.compute_features(features_delta_1) - fourierbasis.compute_features(features)) / 0.00000001
        approxgrad_2 = (fourierbasis.compute_features(features_delta_2) - fourierbasis.compute_features(features)) / 0.00000001
        realgrad = fourierbasis.compute_gradient(features)
        np.testing.assert_almost_equal(realgrad, np.vstack((approxgrad_1, approxgrad_2)).T, decimal=6)

