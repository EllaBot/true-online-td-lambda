import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from true_online_td_lambda import TrueOnlineTDLambda

def plot_value_function(learner):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = y = np.arange(0, 1.0, 0.05)
    X, Y = np.meshgrid(x, y)
    zs = np.array([learner.value([x, y]) for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=True)

    ax.set_xlabel('Feature One')
    ax.set_ylabel('Feature Two')
    ax.set_zlabel('Value')

    plt.draw()