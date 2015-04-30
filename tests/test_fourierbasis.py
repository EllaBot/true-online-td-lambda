from true_online_td_lambda import FourierBasis

from nose.tools import assert_equal
from nose.tools import assert_list_equal

class TestFourierBasis(object):

    def test_get_num_basis_functions(self):
        ranges = None
        n = 2
        d = 3
        fourierbasis = FourierBasis(ranges, n, d)
        assert_equal(fourierbasis.get_num_basis_functions(), 16.0)

    #def test_multipliers(self):
    #    n = 2
    #    d = 3
    #    multipliers = fourierbasis.__multipliers(n, d).tolist()
    #    assert_list_equal(multipliers, [0, 10, 1, 11])


