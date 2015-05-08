from true_online_td_lambda.optimization import brute

import numpy as np

class TestBrute(object):

    def test_maximize(self):
        func = lambda x: -np.power(x + 2, 2).sum()
        bounds = [[-4.0, 4.0], [-4.0, 4.0]]
        x_max = brute.maximize(func, bounds=bounds)
        np.testing.assert_almost_equal(x_max, (-2.0, -2.0))
