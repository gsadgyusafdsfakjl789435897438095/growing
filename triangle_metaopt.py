import numpy as np
import growing_4 as grow
import pickle
import time

if __name__== "__main__":
    # dict of parameters for model from experiment
    # dt is 20 minutes and it is our new time constant
    #parameters = {'a':0.0014, 'b':0.175, 'e1':1, 'e2':10, 'k':0.15, 'dt':1}

    # make data as list of dicts
    p1 = {'a':0.0014, 'b':0.087, 'e1':1, 'e2':1, 'k':0.15, 'dt':1}
    p2 = {'a':0.00055, 'b':0.087, 'e1':1, 'e2':10, 'k':0.15, 'dt':1}
    p3 = {'a':0.0014, 'b':0.087, 'e1':1, 'e2':10, 'k':0.15, 'dt':1}
    p4 = {'a':0.00055, 'b':0.087, 'e1':1, 'e2':1, 'k':0.15, 'dt':1}
    p5 = {'a':0.0014, 'b':0.175, 'e1':1, 'e2':1, 'k':0.15, 'dt':1}
    p6 = {'a':0.00055, 'b':0.175, 'e1':1, 'e2':10, 'k':0.15, 'dt':1}
    p7 = {'a':0.0014, 'b':0.175, 'e1':1, 'e2':10, 'k':0.15, 'dt':1}
    p8 = {'a':0.00055, 'b':0.175, 'e1':1, 'e2':1, 'k':0.15, 'dt':1}
    data = [p3, p4, p5, p6, p7, p8]
    j=3
    # start work
    # growing triangle method metaoptimization
    for parameters in data:
        
        start = time.time()
        #create model of crop
        plant = grow.plant_model(**parameters)
        # fix a cose big boss say this
        a = 1
        betas = np.arange(0.1, 10, 1)
        gammas = np.arange(0.1, 10, 1)
        errors = []
        i = 0
        for b in betas:
            for g in gammas:
                print('iteration number {} beta is {} gamma is {}'.format(i, b, g))
                error = plant.find_triangle_minimum(max_iteration_number = 25,show = False,\
                alpha = a, beta = b, gamma = g)
                # just save all error data for science
                errors.append(error)
                i+=1
        # save results
        end = time.time()
        results = {'data':parameters, 'errors':errors, 'time':(end-start)}
        with open('triangle_metaoptimize_{}.pickle'.format(j), 'wb') as f:
            pickle.dump(results, f)
        with open('triangle_metaoptimize_double{}.pickle'.format(j), 'wb') as f:
            pickle.dump(results, f)
        j+=1
