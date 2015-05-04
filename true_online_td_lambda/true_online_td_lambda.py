from basis import FourierBasis
from optimization import l_bfgs
import numpy as np
from random import random


class TrueOnlineTDLambda(object):
    def __init__(self, numfeatures, ranges):
        self.alpha = 0.01
        self.lmbda = 0.9
        self.gamma = 0.99
        self.basis = FourierBasis(ranges, numfeatures, 3)
        self.stateprime = None
        self.state = None
        self.theta = np.zeros(self.basis.get_num_basis_functions())
        self.vs = 0
        self.traces = np.zeros(self.basis.get_num_basis_functions())

    def step(self, reward, state):
        # Rotate the states back
        self.state = self.stateprime
        self.stateprime = state

        # If we don't have two states, there will not be an update
        if self.state is None:
            return

        phi_t = self.basis.compute_features(self.state)
        phi_tp = self.basis.compute_features(self.stateprime)

        vsprime = np.dot(self.theta, phi_tp)

        # See "True Online TD(lambda)" section 4.1
        delta = reward + (self.gamma * vsprime) - self.vs
        termone = self.gamma * self.lmbda * self.traces
        termtwo = self.alpha * (1 - self.gamma * self.lmbda * np.dot(self.traces, phi_t)) * phi_t
        self.traces = termone + termtwo
        self.theta += delta * self.traces + self.alpha * (vsprime - np.dot(self.theta, phi_t)) * phi_t

        self.vs = vsprime

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
        f = lambda x: self.value(np.concatenate((state, x)))
        # Compute the gradient, and only select the column(s) which has partial derivatives w.r.t the actions
        fprime = lambda x: np.dot(self.theta, self.basis.compute_gradient(np.concatenate((state, x)))[:, len(state):])

        # Initial guess is the midpoint of range
        initial_guess = np.array([(float(self.basis.ranges[i][1]) + float(self.basis.ranges[i][0])) / 2.0 for i in range(len(state), len(self.basis.ranges))])

        return maximize(f, initial_guess, fprime)
