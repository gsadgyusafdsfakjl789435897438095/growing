import numpy as np
import growing_4 as grow
import pylab as pl
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import cm
import matplotlib.patches
import matplotlib.path

if __name__== "__main__":
    # dict of parameters for model from experiment
    # dt is 20 minutes and it is our new time constant
    parameters = {'a':0.0014, 'b':0.175, 'e1':1, 'e2':10, 'k':0.15, 'dt':1}
    plant = grow.plant_model(**parameters)
    # growing gradient method metaoptimization
    gammas = np.arange(0.01, 2.1, 0.1)
    print(np.shape(gammas))
    errors = []
    for i in range(len(gammas)):
        print('iteration number {}'.format(i))
        error = plant.find_gradient_minimum(max_iteration_number = 50,show = False, gamma = gammas[i])
        errors.append(np.mean(error))
    # show squared errors by steps
    fig = pl.figure()
    pl.plot(gammas, errors, "-vr")
    pl.grid()
    pl.savefig("gradient_metaopt_1.png")
    pl.show()