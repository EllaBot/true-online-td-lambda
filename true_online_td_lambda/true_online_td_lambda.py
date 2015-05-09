from basis import FourierBasis
from optimization import l_bfgs, brute
import numpy as np
import math
from random import random


class TrueOnlineTDLambda(object):
    def __init__(self, numfeatures, ranges, alpha=0.0001, lmda=0.9, gamma=0.99):

        """

        :param numfeatures: the number of items in a state-action pair
        :param ranges: an array of tuples representing the closed intervals of
                         features
        :param alpha: the learning rate
        :param lmda: the rate at which to decay the traces
        :param gamma: the discount factor governing how future rewards
                          are weighted with respect to the immediate reward
        """
        assert 0 <= lmda <= 1
        assert len(ranges) is numfeatures
        self.alpha = alpha
        self.lmbda = lmda
        self.gamma = gamma
        self.basis = FourierBasis(ranges, numfeatures, 3)
        self.stateprime = None
        self.state = None
        self.theta = np.zeros(self.basis.get_num_basis_functions())
        self.vs = 0
        self.traces = np.zeros(self.basis.get_num_basis_functions())

    def start(self, state):
        """Provide the learner with the first state.
        """
        self.stateprime = state
        self.vs = self.value(state)


    def step(self, reward, state):
        """ Perform the typical update
        See "True Online TD(lambda)" section 4.1
        """
        # Rotate the states back
        self.state = self.stateprime
        self.stateprime = state

        assert self.state is not None and self.stateprime is not None
        phi_t = self.basis.compute_features(self.state)
        vsprime = self.value(self.stateprime)

        self.updatetraces(phi_t)
        self.updateweights(phi_t, reward, self.vs, vsprime)

        self.vs = vsprime

    def updatetraces(self, phi_t):
        """Compute the e vector.

        :param phi_t: vector of values of the features of the initial state.
                        Parameterized so it can be cached between calls.
        """
        termone = self.gamma * self.lmbda * self.traces
        termtwo = self.alpha * (1 - self.gamma * self.lmbda * np.dot(self.traces, phi_t)) * phi_t
        self.traces = termone + termtwo

    def updateweights(self, phi_t, reward, vs, vsprime):
        """Compute the theta vector.

        :param phi_t: vector of values of the features of the initial state.
        :param reward: the reward signal received
        :param vs: the value of the state
        :param vsprime: the value of the resulting state
        """
        delta = reward + (self.gamma * vsprime) - vs
        self.theta += delta * self.traces + self.alpha * (vs - np.dot(self.theta, phi_t)) * phi_t

    def end(self, reward):
        """ Receive the reward from the final action.

        This action does not produce an additional state, so we update a
        little differently.
        """
        if self.state is None:
            # If we're ending before we have a state, we
            # don't have enough data to perform an update. Just reset.
            self._reset()
            return
        phi_t = self.basis.compute_features(self.state)
        # There is no phi_tp because there is no second state, so we'll
        # set the value of the second state to zero.
        vsprime = np.zeros(self.traces.shape)

        self.updatetraces(phi_t)
        self.updateweights(phi_t, reward, self.vs, vsprime)

        # Clear episode specific learning artifacts
        self._reset()

    def value(self, state_action):
        """Compute the value with the current weights.

        :param state_action:
        :return: The value as a float.
        """
        value = np.dot(self.theta, self.basis.compute_features(state_action))

        # If we've diverged, just crash
        assert not math.isnan(value) and not math.isinf(value)
        return value

    def maximize_value(self, state, maximize=brute.maximize):
        """Maximize the value function w.r.t the features

        Parameters
        ----------
        state: list
            List of values of the state

        maximize: function, optional
            Maximization function, l_bfgs by default
        """
        # Scale state
        state_scaled = self.basis.scale_features(state)

        f = lambda x: np.dot(self.theta, self.basis.compute_scaled_features(np.append(state_scaled, x)))
        # Compute the gradient, and only select the column(s) which has partial derivatives w.r.t the actions
        fprime = lambda x: np.dot(self.theta, self.basis.compute_gradient(np.concatenate((state_scaled, x)))[:, len(state):])

        # Initial guess is the midpoint of range
        initial_guess = self.basis.scale_features([(float(self.basis.ranges[i][1]) + float(self.basis.ranges[i][0])) / 2.0 for i in range(len(state), len(self.basis.ranges))])

        # Bounds
        bounds = [[0.0, 1.0]] * (len(self.basis.ranges) - len(state))

        maximum = maximize(f, x0=initial_guess, func_prime=fprime, bounds=bounds)

        # Unscale features
        ranges = self.basis.ranges[-len(maximum):]
        return maximum * (ranges[:, 1] - ranges[:, 0]) + ranges[:, 0]

    def _reset(self):
        self.state = None
        self.stateprime = None
        self.vs = None
        self.traces.fill(0.0)
