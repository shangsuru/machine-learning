import numpy as np
from math import pi, sqrt, exp, log

"""
parameter list as follows
[[expectation, covariance, weight], ...]

"""

data = np.loadtxt('dataSets/gmm.txt')

def log_likelihood(data, px):
    sum = 0
    for sample in data:
        sum += log(px(sample))
    return sum

def multi_gaussian(x, ex, cov):
    D = x.shape[0]
    return (1/((2 * pi) ** (D / 2))) * (1 / sqrt(np.linalg.det(cov))) * exp(-0.5 * ((x-ex).T@np.linalg.inv(cov)@(x-ex)))


def mix_of_gaussian(x, parameters):
    sum = 0
    for component in parameters:
        sum += multi_gaussian(x, component[0], component[1]) * component[2]
    return sum


def estimation_maximization(data, init_parameters):
    

    ##### estep
    def e_step():
        posteriors = []
        Njs = []
        for i in range(len(parameters)):
            Njs.append(0)
        for x in data:
            posterior = []
            sum_comps = 0
            for component in parameters:
                post = component[2] * multi_gaussian(x, component[0], component[1])
                posterior.append(post)
                sum_comps += post
            for i in range(len(posterior)):
                posterior[i] = posterior[i] / sum_comps
                Njs[i] += posterior[i]
            posteriors.append(posterior)
        return posteriors, Njs
    
    ##### mstep
    def  m_step():
        for index, component in enumerate(parameters):
            new_ex = 0
            for i in range(len(data)):
                new_ex += posteriors[i][index] * data[i]
            new_ex = new_ex / Njs[index]
            parameters[index][0] = new_ex
            
            new_cov = 0

            for i in range(len(data)):
                x = (data[i] - new_ex)
                x.shape = (2, 1)
                t = x.T
                new_cov += x@t * posteriors[i][index]
            new_cov = new_cov / Njs[1]
            if np.linalg.det(new_cov) > 0:
                parameters[index][1] = new_cov
            parameters[index][2] = Njs[1] / sum(Njs)
        


    parameters = init_parameters
    posteriors = []
    Njs = []
    px = lambda x: mix_of_gaussian(x, parameters)

    prev_parameters = parameters[:]
    prev_ll = log_likelihood(data, px)
    ii = 0
    while True:
        posteriors, Njs = e_step()
        m_step()
        px = lambda x: mix_of_gaussian(x, parameters)
        ll = log_likelihood(data, px)
        #if ii <= 30:
        print('iteration ' + str(ii) + ': ' + str(prev_ll))
        ii += 1
        if abs(prev_ll - ll) < 1e-10:
            break
        prev_ll = ll
    
    return prev_parameters



testparam = [
    [[2, 2], [[1, 0], [0, 1]], 0.25],
    [[0.5, 1], [[1, 0], [0, 1]], 0.25],
    [[0.5, 1.5], [[1, 0], [0, 1]], 0.25],
    [[1, 0], [[1, 0], [0, 1]], 0.25]
]

print(estimation_maximization(data, testparam))

