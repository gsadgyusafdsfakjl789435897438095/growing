import numpy as np
import pylab as pl
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import cm
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
    B, G = np.meshgrid(betas, gammas)
    d1b, d2b = np.shape(B)
    d1g, d2g = np.shape(G)
    B_ = B.reshape((d1b*d2b, 1))
    G_ = G.reshape((d1g*d2g, 1))
    #with open('triangle_metaoptimize soft_2.pickle', 'rb') as f:777
    with open('triangle_metaoptimize_2.pickle', 'rb') as f:
        res = pickle.load(f)
    #errors = np.zeros((len(gammas))*(len(betas)))
    e = res['errors']
    yaw_errors = np.zeros(len(e))
    find_errors = np.zeros(len(e))
    for i in range(len(e)):
        #yaw_errors[i] = (np.mean(e[i][15:len(e[i])]*e[i][15:len(e[i])]))**0.5
        #find_errors[i] = (np.mean(e[i][0:15]*e[i][0:15]))**0.5
        yaw_errors[i] = np.mean(e[i][15:len(e[i])])
        find_errors[i] = np.mean(e[i][0:15])
    #print((errors))
    # min
    min_yaw_num = np.argmin(yaw_errors)
    interesting_yaw  = e[min_yaw_num]
    print("min yaw error is {}".format(yaw_errors[min_yaw_num]))
    min_find_num = np.argmin(find_errors)
    interesting_find  = e[min_find_num]
    print("min find error is {}".format(yaw_errors[min_find_num]))
    #interesting_yaw  = e[min_yaw_num]
    # max
    max_yaw_num = np.argmax(yaw_errors)
    interesting_max_yaw  = e[max_yaw_num]
    print("max yaw error is {}".format(yaw_errors[max_yaw_num]))
    max_find_num = np.argmax(find_errors)
    interesting_max_find  = e[max_find_num]
    print("max find error is {}".format(find_errors[max_find_num]))
    #interesting_max_yaw  = e[max_yaw_num]
    # 2D plot of min yaw errors
    fig=pl.figure()
    pl.plot(range(0, len(interesting_yaw)), interesting_yaw, '-or')
    pl.ylabel('Yaw min error')
    pl.xlabel('time')
    pl.title("Minimal yaw errors for all time in \n {}".format(str(res['data'])))
    #ax_.set_zlabel('Squared mean error')
    pl.grid()
    pl.show()
    # 2D plot of max yaw errors
    fig=pl.figure()
    pl.plot(range(0, len(interesting_max_yaw)), interesting_max_yaw, '-vb')
    pl.ylabel('Yaw max error')
    pl.xlabel('time')
    pl.title("Max yaw errors for all time in \n {}".format(str(res['data'])))
    #ax_.set_zlabel('Squared mean error')
    pl.grid()
    pl.show()
    
    E_yaw = yaw_errors.reshape(len(gammas),(len(betas)))
    E_find = find_errors.reshape(len(gammas),(len(betas)))
    # 3D surface plot of yaw errors
    fig = pl.figure()
    ax = p3.Axes3D(fig)
    cs = ax.plot_surface(B, G, E_yaw, rstride = 1, cstride = 1, color = 'g', cmap=cm.coolwarm)
    pl.clabel(cs, fmt = '%.1f', colors="black")
    fig.colorbar(cs, shrink=0.5, aspect=5)
    ax.set_ylabel('Beta')
    ax.set_xlabel('Gamma')
    ax.set_zlabel('Mean yaw error')
    pl.grid()
    #pl.savefig("gradient_metaopt_5678676787656765456765.png")
    pl.show()
    
    # 3D surface plot of find errors
    fig = pl.figure()
    ax = p3.Axes3D(fig)
    cs = ax.plot_surface(B, G, E_find, rstride = 1, cstride = 1, color = 'g', cmap=cm.coolwarm)
    pl.clabel(cs, fmt = '%.1f', colors="black")
    fig.colorbar(cs, shrink=0.5, aspect=5)
    ax.set_ylabel('Beta')
    ax.set_xlabel('Gamma')
    ax.set_zlabel('Mean find error')
    pl.grid()
    #pl.savefig("gradient_metaopt_5678676787656765456765.png")
    pl.show()
    
    # 2D proection plot
    fig_=pl.figure()
    #ax_ = p3.Axes3D(fig_)
    cs_ = pl.contour(B, G, E_yaw, 20)
    pl.clabel(cs_)
    pl.ylabel('Beta')
    pl.xlabel('Gamma')
    pl.title("Mean yaw error for \n {}".format(str(res['data'])))
    #ax_.set_zlabel('Squared mean error')
    pl.grid()
    # Add a color bar which maps values to colors.
    fig_.colorbar(cs_, shrink=0.5, aspect=5)

    pl.show()
    
    #with open('triangle_metaopt.pickle', 'rb') as f:
        #errors = pickle.load(f)
    # show squared errors by steps
    
    #E3 = errors4.reshape(len(gammas),(len(betas)))
    #print(errors)
    #print(np.shape(errors))
    #print(np.shape(E))
    #print(np.shape(B))
    #print(np.shape(G))
    #fig = pl.figure()
    #ax = p3.Axes3D(fig)
    #pl.plot(gammas, errors, "-vr")
    #cs = ax.plot_surface(B, G, E, rstride = 1, cstride = 1, color = 'g', cmap=cm.coolwarm)
    #fig = pl.figure()
    #ax = p3.Axes3D(fig)
    #cs = ax.plot_surface(B, G, E, rstride = 1, cstride = 1, color = 'g', cmap=cm.coolwarm)
    ###pl.clabel(cs, fmt = '%.1f', colors="black")
    #pl.clabel(cs, fmt = '%.1f', colors="black")
    ###fig.colorbar(cs, shrink=0.5, aspect=5)
    ### Add a color bar which maps values to colors.
    #fig.colorbar(cs, shrink=0.5, aspect=5)
    #pl.grid()
    ##pl.savefig("gradient_metaopt_5678676787656765456765.png")
#
    #pl.show()
