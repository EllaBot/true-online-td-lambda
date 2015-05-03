from true_online_td_lambda import TrueOnlineTDLambda
from nose.tools import assert_greater


class TestTrueOnlineTDLambda(object):

    def test_learn_relative_value(self):
        state_one = [0, 0]
        state_two = [1, 1]
        state_three = [0.5, 0.5]
        learner = TrueOnlineTDLambda(2, [(0, 1), (0, 1)])

        learner.step(100, [0, 1])
        for x in xrange(0, 100):
            learner.step(10, state_two)
        for x in xrange(0, 100):
            learner.step(-10, state_one)

        assert_greater(learner.value(state_two), learner.value(state_one))
        assert_greater(learner.value(state_two), learner.value(state_three))
        assert_greater(learner.value(state_three), learner.value(state_one))