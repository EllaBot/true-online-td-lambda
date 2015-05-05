from basis import FourierBasis
from optimization import l_bfgs
import numpy as np
from random import random


class TrueOnlineTDLambda(object):
    def __init__(self, numfeatures, ranges, alpha = 0.01, lmda = 0.9, gamma = 0.99):
        self.alpha = alpha
        self.lmbda = lmda
        self.gamma = gamma
        self.basis = FourierBasis(ranges, numfeatures, 3)
        self.stateprime = None
        self.state = None
        self.theta = np.zeros(self.basis.get_num_basis_functions())
        self.vs = 0
        self.traces = np.zeros(self.basis.get_num_basis_functions())

    """The learner cannot perform an update without having an
        initial state.
    """
    def start(self, state):
        self.stateprime = state
        self.vs = self.value(state)

    """ See "True Online TD(lambda)" section 4.1
    """
    def step(self, reward, state):
        # Rotate the states back
        self.state = self.stateprime
        self.stateprime = state

        assert self.state is not None and self.stateprime is not None
        phi_t = self.basis.compute_features(self.state)
        phi_tp = self.basis.compute_features(self.stateprime)
        vsprime = np.dot(self.theta, phi_tp)

        self.updatetraces(phi_t)
        self.updateweights(phi_t, reward, self.vs, vsprime)

        self.vs = vsprime

    def updatetraces(self, phi_t):
        termone = self.gamma * self.lmbda * self.traces
        termtwo = self.alpha * (1 - self.gamma * self.lmbda * np.dot(self.traces, phi_t)) * phi_t
        self.traces = termone + termtwo

    def updateweights(self, phi_t, reward, vs, vsprime):
        delta = reward + (self.gamma * vsprime) - vs
        self.theta += delta * self.traces + self.alpha * (vs - np.dot(self.theta, phi_t)) * phi_t

    """ Receive the reward from the final action.
        This action does not produce an additional state, so we update a little differently.
    """
    def end(self, reward):
        phi_t = self.basis.compute_features(self.state)
        # There is no phi_tp because there is no second state, so we'll
        # set the value of the second state to zero.
        vsprime = np.zeros(self.traces.shape)

        self.updatetraces(phi_t)
        self.updateweights(phi_t, reward, self.vs, vsprime)

        # Clear episode specific learning artifacts
        self._reset()

    def value(self, state_action):
        return np.dot(self.theta, self.basis.compute_features(state_action))

    def maximize_value(self, state, maximize = l_bfgs.maximize):
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

        f = lambda x: np.dot(self.theta, self.basis.compute_scaled_features(np.concatenate((state_scaled, x))))
        # Compute the gradient, and only select the column(s) which has partial derivatives w.r.t the actions
        fprime = lambda x: np.dot(self.theta, self.basis.compute_gradient(np.concatenate((state_scaled, x)))[:, len(state):])

        # Initial guess is the midpoint of range
        initial_guess = self.basis.scale_features([(float(self.basis.ranges[i][1]) + float(self.basis.ranges[i][0])) / 2.0 for i in range(len(state), len(self.basis.ranges))])

        # Bounds
        bounds = [[0.0, 1.0]] * (len(self.basis.ranges) - len(state))

        maximum = maximize(f, initial_guess, fprime, bounds=bounds)

        # Unscale features
        ranges = self.basis.ranges[-len(maximum):]
        return maximum * (ranges[:, 1] - ranges[:, 0]) + ranges[:, 0]

    def _reset(self):
        self.state = None
        self.stateprime = None
        self.vs = None
        self.traces.fill(0.0)
