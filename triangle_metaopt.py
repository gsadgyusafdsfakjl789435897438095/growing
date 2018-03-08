import numpy as np
import growing_4 as grow
import pylab as pl
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import cm
import matplotlib.patches
import matplotlib.path
import pickle

if __name__== "__main__":
    # dict of parameters for model from experiment
    # dt is 20 minutes and it is our new time constant
    parameters = {'a':0.0014, 'b':0.175, 'e1':1, 'e2':10, 'k':0.15, 'dt':1}
    parameters2 = {'a':0.0014, 'b':0.175, 'e1':1, 'e2':1, 'k':0.15, 'dt':1}
    parameters3 = {'a':0.14, 'b':0.175, 'e1':1, 'e2':10, 'k':0.15, 'dt':1}
    parameters4 = {'a':0.0014, 'b':0.0175, 'e1':1, 'e2':1, 'k':0.15, 'dt':1}
    plant = grow.plant_model(**parameters)
    # growing gradient method metaoptimization
    a = 1
    betas = np.arange(0.1, 10, 1)
    gammas = np.arange(0.1, 10, 1)
    ## mesh data
    #B, G = np.meshgrid(betas, gammas)
    #d1b, d2b = np.shape(B)
    #d1g, d2g = np.shape(G)
    #B_ = B.reshape((d1b*d2b, 1))
    #G_ = G.reshape((d1g*d2g, 1))
    #print(np.shape(gammas))
    errors = np.zeros((len(betas)*len(gammas)))
    i = 0
    for b in betas:
        for g in gammas:
            print('iteration number {}'.format(i))
            error = plant.find_triangle_minimum(max_iteration_number = 25,show = False,\
            alpha = a, beta = b, gamma = g)
            errors[i] = np.mean(error)
            i+=1
    # save results
    with open('triangle_metaopt6.pickle', 'wb') as f:
        pickle.dump(errors, f)
    with open('triangle_metaopt_double6.pickle', 'wb') as f:
        pickle.dump(errors, f)
    #with open('triangle_metaopt2.pickle', 'rb') as f:
        #errors = pickle.load(f)
    ## show squared errors by steps
    #E = errors.reshape((len(gammas), len(betas)))
    #fig = pl.figure()
    #ax = p3.Axes3D(fig)
    ##pl.plot(gammas, errors, "-vr")
    #cs = ax.plot_surface(B, G, E , rstride = 3, cstride = 3, color = 'g', cmap=cm.coolwarm)
    #pl.clabel(cs, fmt = '%.1f', colors="black")
    ## Add a color bar which maps values to colors.
    #fig.colorbar(cs, shrink=0.5, aspect=5)
    #pl.grid()
    #pl.savefig("triangle_metaopt_2.png")
    #pl.show()