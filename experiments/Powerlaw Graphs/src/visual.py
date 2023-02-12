import matplotlib.cm as cm
from matplotlib import pyplot as plt

def matrix_to_fig(z, name):
    # plt.imshow(z, cmap=cm.Set1)
    plt.imshow(z, cmap=cm.Blues)
    cb = plt.colorbar(orientation='horizontal', shrink=.75)
    cb.set_label('colormaping')
    # plt.show()
    plt.savefig(name + '.pdf')
    plt.clf()

