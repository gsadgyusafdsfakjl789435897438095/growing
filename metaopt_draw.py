import numpy as np
import pylab as pl
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import cm
import matplotlib.patches
import matplotlib.path
import pickle


if __name__== "__main__":
    #betas = np.arange(0.4, 0.6, 0.02)
    #gammas = np.arange(2.8, 3, 0.02)
    #betas = np.arange(120, 140, 1)
    #gammas = np.arange(270, 290, 1)
    betas = np.arange(0.1, 10, 1)
    gammas = np.arange(0.1, 10, 1)
    # mesh data
    B, G = np.meshgrid(betas, gammas)
    d1b, d2b = np.shape(B)
    d1g, d2g = np.shape(G)
    B_ = B.reshape((d1b*d2b, 1))
    G_ = G.reshape((d1g*d2g, 1))
    with open('triangle_metaopt6.pickle', 'rb') as f:
        errors4 = pickle.load(f)
    #with open('triangle_metaopt.pickle', 'rb') as f:
        #errors = pickle.load(f)
    # show squared errors by steps
    #E = errors.reshape(len(gammas),(len(betas)))
    E3 = errors4.reshape(len(gammas),(len(betas)))
    #print(errors)
    #print(np.shape(errors))
    #print(np.shape(E))
    #print(np.shape(B))
    #print(np.shape(G))
    #fig = pl.figure()
    #ax = p3.Axes3D(fig)
    #pl.plot(gammas, errors, "-vr")
    #cs = ax.plot_surface(B, G, E, rstride = 1, cstride = 1, color = 'g', cmap=cm.coolwarm)
    fig2 = pl.figure()
    ax2 = p3.Axes3D(fig2)
    cs2 = ax2.plot_surface(B, G, E3, rstride = 1, cstride = 1, color = 'g', cmap=cm.coolwarm)
    #pl.clabel(cs, fmt = '%.1f', colors="black")
    pl.clabel(cs2, fmt = '%.1f', colors="black")
    #fig.colorbar(cs, shrink=0.5, aspect=5)
    # Add a color bar which maps values to colors.
    fig2.colorbar(cs2, shrink=0.5, aspect=5)
    pl.grid()
    pl.savefig("gradient_metaopt_4.png")

    pl.show()