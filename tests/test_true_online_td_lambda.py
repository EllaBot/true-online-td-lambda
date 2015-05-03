from true_online_td_lambda import TrueOnlineTDLambda
from nose.tools import assert_greater, assert_equal, assert_almost_equal
from numpy.testing import assert_array_equal
from true_online_td_lambda import learner_plotting_utilities

class TestTrueOnlineTDLambda(object):

    def test_learn_relative_value(self):
        state_one = [0, 0]
        state_two = [1, 1]
        state_three = [0.5, 0.5]
        learner = TrueOnlineTDLambda(2, [(0, 1), (0, 1)])

        learner.step(100, [0, 1])
        for x in range(0, 100):
            learner.step(10, state_two)
        for x in range(0, 100):
            learner.step(-10, state_one)

        assert_greater(learner.value(state_two), learner.value(state_one))
        assert_greater(learner.value(state_two), learner.value(state_three))
        assert_greater(learner.value(state_three), learner.value(state_one))

    def test_learning_scaled(self):
        state_one = [0.0, 0.0]
        learner = TrueOnlineTDLambda(2, [(0, 1), (0, 1)])
        r = 20
        for i in range(100):
            learner.step(-r, state_one)

        for i in range(100):
            learner.step(r, [1.0, 1.0])


        learner_scaled = TrueOnlineTDLambda(2, [(0, 20.0), (0, 20.0)])
        for i in range(100):
            learner_scaled.step(-r, state_one)

        for i in range(100):
            learner_scaled.step(r, [20.0, 20.0])

        print(learner.theta)
        print(learner_scaled.theta)
        assert_array_equal(learner.theta, learner_scaled.theta)
        assert_almost_equal(learner.value([1.0, 1.0]), learner_scaled.value([20.0, 20.0]))
        assert_almost_equal(learner.value(state_one), learner_scaled.value(state_one))
