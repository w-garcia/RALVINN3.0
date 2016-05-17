import pylab as pl


def plot_weights(weights):

    [_, nodes] = weights.shape
    actions = ['left', 'right']

    pl.figure()
    for node in range(nodes):
        pl.subplot(1, nodes, node + 1)
        pl.imshow(weights[:, node].reshape((4, 3)), interpolation='nearest')
        pl.title("Weights for action " + actions[node])
    pl.show()
