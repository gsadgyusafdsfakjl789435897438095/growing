import numpy as np
import pylab as pl
#import mpl_toolkits.mplot3d.axes3d as p3
#from matplotlib import cm
#import matplotlib.patches
#import matplotlib.path
import pickle


if __name__== "__main__":
    
    #rude triangle
    betas = np.arange(0.1, 10, 1)
    gammas = np.arange(0.1, 10, 1)
    #soft triangle
    #betas = np.arange(0.01, 2, 0.1)
    #gammas = np.arange(1, 5, 0.5)
    #giant triangle
    #betas = np.arange(0.1, 100, 9)
    #gammas = np.arange(0.1, 100, 9)
    # mesh data
    #B, G = np.meshgrid(betas, gammas)
    #d1b, d2b = np.shape(B)
    #d1g, d2g = np.shape(G)
    #B_ = B.reshape((d1b*d2b, 1))
    #G_ = G.reshape((d1g*d2g, 1))
    gammas = np.arange(0.001, 2, 0.05)
    with open('gradient_metaoptimize_1.pickle', 'rb') as f:
        res = pickle.load(f)
    #errors = np.zeros((len(gammas))*(len(betas)))
    e = res['errors']
    errors = np.zeros(len(e))
    for i in range(len(e)):
        errors[i] = (np.mean(e[i]*e[i]))**0.5
    print((errors))
    E = errors.reshape(len(gammas))
    ## 3D surface plot
    #fig = pl.figure()
    #ax = p3.Axes3D(fig)
    #cs = ax.plot_surface(B, G, E, rstride = 1, cstride = 1, color = 'g', cmap=cm.coolwarm)
    #pl.clabel(cs, fmt = '%.1f', colors="black")
    #fig.colorbar(cs, shrink=0.5, aspect=5)
    #ax.set_ylabel('Beta')
    #ax.set_xlabel('Gamma')
    #ax.set_zlabel('Squared mean error')
    #pl.grid()
    ##pl.savefig("gradient_metaopt_5678676787656765456765.png")
    #pl.show()
    
    # 2D proection plot
    fig_=pl.figure()
    #ax_ = p3.Axes3D(fig_)
    pl.plot(gammas, errors, '-rv')
    #pl.clabel(cs_)
    #pl.ylabel('Error')
    #pl.xlabel('Gamma')
    pl.title("Squared mean error for \n {}".format(str(res['data'])))
    #ax_.set_zlabel('Squared mean error')
    pl.grid()
    # Add a color bar which maps values to colors.
    #fig_.colorbar(cs_, shrink=0.5, aspect=5)

    pl.show()

