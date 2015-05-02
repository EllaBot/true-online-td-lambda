from fourierbasis import FourierBasis
import numpy as np
from random import random
class TrueOnlineTDLamda(object):
    def __init__(self, numfeatures, ranges):
        self.alpha = 0.01
        self.epsilon = 0.1
        self.lmbda = 0.7
        self.gamma = 1.0
        self.basis = FourierBasis(ranges, numfeatures)
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
        termone = self.gamma * self.lmbda * self.traces + self.alpha
        termtwo = 1 - self.gamma * self.lmbda * np.dot(self.traces, phi_t) * phi_t
        self.traces = termone * termtwo
        self.theta += delta * self.traces + self.alpha * (vsprime - np.dot(self.theta, phi_t)) * phi_t

        self.vs = vsprime

    def egreedy(self):
        if random() < self.epsilon:
            # Produce a random action
            return
        # Otherwise, find the best possible action to take from the current state