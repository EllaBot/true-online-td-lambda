from true_online_td_lambda import TrueOnlineTDLambda
from true_online_td_lambda.optimization import l_bfgs, brute
import math
from nose.tools import assert_greater, assert_equal, assert_almost_equal
from numpy.testing import assert_array_equal

class TestTrueOnlineTDLambda(object):

    def test_learn_relative_value(self):
        state_one = [0, 0]
        state_two = [1, 1]
        state_three = [0.5, 0.5]
        learner = TrueOnlineTDLambda(2, [(0, 1), (0, 1)])

        learner.start([0, 1])
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
        learner.start(state_one)
        for i in range(100):
            learner.step(-r, state_one)

        for i in range(100):
            learner.step(r, [1.0, 1.0])

        learner_scaled = TrueOnlineTDLambda(2, [(0, 20.0), (0, 20.0)])
        learner_scaled.start(state_one)
        for i in range(100):
            learner_scaled.step(-r, state_one)

        for i in range(100):
            learner_scaled.step(r, [20.0, 20.0])

        # The problem are in fact, identical. Theta should not be affected by arbitrary scales.
        assert_array_equal(learner.theta, learner_scaled.theta)
        assert_almost_equal(learner.value([1.0, 1.0]), learner_scaled.value([20.0, 20.0]))
        assert_almost_equal(learner.value(state_one), learner_scaled.value(state_one))

    def test_maximize_value(self):
        learner = TrueOnlineTDLambda(2, [(0, 10), (0, 10)])
        learner.start([0, 0])
        for n in range(100):
            i = n / 10.0
            diff = i - 7.6  # 7.6 is the target
            learner.step(-math.sqrt(diff ** 2 + diff ** 2), [float(i), float(i)])

        state_1 = 5.0  # fix state one
        max_action = learner.maximize_value([state_1], maximize=l_bfgs.maximize)

        print(max_action)

        assert_greater(learner.value([state_1, max_action[0]]), learner.value([state_1, max_action[0] - 0.01]))
        assert_greater(learner.value([state_1, max_action[0]]), learner.value([state_1, max_action[0] + 0.01]))

        max_action = learner.maximize_value([state_1], maximize=brute.maximize)

        print(max_action)

        assert_greater(learner.value([state_1, max_action[0]]), learner.value([state_1, max_action[0] - 0.01]))
        assert_greater(learner.value([state_1, max_action[0]]), learner.value([state_1, max_action[0] + 0.01]))

    def test_maximize_value_no_setup(self):
        learner = TrueOnlineTDLambda(2, [(0, 10), (0, 10)])

        max_action = learner.maximize_value([0.0], maximize=l_bfgs.maximize)
        max_action_two = learner.maximize_value([7.6], maximize=l_bfgs.maximize)

        print(max_action)

        # Regardless of the state, the maximum should be the midpoint of the range
        assert_equal(max_action, 5.0)
        assert_equal(max_action_two, 5.0)

    def test_four_dimensions(self):
        learner = TrueOnlineTDLambda(4, [(-1, 1), (-1, 1), (0, 100), (-1, 1)])

        learner.start([0, 0, 100, 1])
        for x in range(0, 100):
            distance = 100 - x
            learner.step(-distance, [0, 0, distance, distance/100])
        learner.end(100)
