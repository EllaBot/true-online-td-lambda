from true_online_td_lambda.optimization import l_bfgs

import numpy as np

class TestLBFGS(object):

    def test_maximize(self):
        func = lambda x: -np.power(x + 2, 2)
        func_grad = lambda x: -2 * (x + 2)
        x0 = np.array([3.0, 2.0])
        x_max = l_bfgs.maximize(func, x0, func_grad)
        np.testing.assert_almost_equal(x_max, (-2.0, -2.0))
