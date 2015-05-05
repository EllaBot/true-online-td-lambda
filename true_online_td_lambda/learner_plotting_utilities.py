import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from true_online_td_lambda import TrueOnlineTDLambda


def plot_two_feature_value_function(learner):
    assert learner.basis.d is 2
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ranges = learner.basis.ranges

    x_range = ranges[0]
    y_range = ranges[1]

    x_resolution = (x_range[1] - x_range[0])/50.0
    y_resolution = (y_range[1] - y_range[0])/50.0

    x = np.arange(x_range[0], x_range[1], x_resolution)
    y = np.arange(y_range[0], y_range[1], y_resolution)
    X, Y = np.meshgrid(x, y)
    zs = np.array([learner.value([x, y]) for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)

    ax.set_xlabel('Feature One')
    ax.set_ylabel('Feature Two')
    ax.set_zlabel('Value')

    plt.draw()


"""
"""
def plot_four_feature_value_function(learner, hold_one, at_value, hold_two, at_value_two):
    assert learner.basis.d is 4
    assert 0 <= hold_one < 4
    assert 0 <= hold_two < 4
    assert hold_one is not hold_two
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ranges = learner.basis.ranges

    x_range = ranges[hold_one]
    y_range = ranges[hold_two]

    x_resolution = (x_range[1] - x_range[0])/50.0
    y_resolution = (y_range[1] - y_range[0])/50.0

    x = np.arange(x_range[0], x_range[1], x_resolution)
    y = np.arange(y_range[0], y_range[1], y_resolution)
    X, Y = np.meshgrid(x, y)
    zs = np.array([learner.value([at_value, at_value_two, x, y]) for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=True)
    ax.set_xlabel('Feature One')
    ax.set_ylabel('Feature Two')
    ax.set_zlabel('Value')

    plt.draw()

def _plot(X,Y,Z):
    pass

def show():
    plt.show()