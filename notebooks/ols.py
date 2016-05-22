import numpy as np
from numpy.linalg import inv

def ols(Y, X, kern):
    result = dict()
    Y = np.array(Y)
    X = np.vstack((np.ones(X.shape[1]), X)).T
    Qxx = np.dot(X.T, X)
    Qxy = np.dot(X.T, Y)
    # Parameter estimate
    beta = np.dot(inv(Qxx), Qxy)
    # Residual estimates
    e = Y - np.dot(X, beta)
    # Estimate of asymptotic variance
    V = HAC(e, X, kern)
    # Corresponding standard errors
    s = np.diag(V) ** .5
    # t-statistics
    t = beta / s
    result = {'beta' : beta, 'V' : V, 's' : s, 't' : t, 'e' : e}
    return result

def HAC(e, X, kern):
    N = X.shape[0]
    q = round(N**(1/5))
    for m in range(0, N):
        G = np.dot(X[m:].T * e[m:], (X[:N-m].T * e[:N-m]).T)
        if m == 0:
            S = G
        else:
            w = kernel(m / q, kern)
            S += w * (G + G.T)
    Q = inv(np.dot(X.T, X))
    V = np.dot(Q, S).dot(Q)
    return V

def kernel(x, name):
    kernels = {'White' : White,
               'HansenHodrick' : HansenHodrick,
               'Bartlett' : Bartlett,
               'Parzen' : Parzen}
    return kernels[name](x)

def White(x):
    return 0

def HansenHodrick(x):
    if abs(x) <= 1:
        return 1
    else:
        return 0

def Bartlett(x):
    if abs(x) <= 1:
        return 1 - abs(x)
    else:
        return 0

def Parzen(x):
    if abs(x) <= .5:
        return 1 - 6 * x**2 + 6 * abs(x)**3
    if abs(x) > .5 and abs(x) <= .5:
        return 3 * (1 - abs(x)**3)
    else:
        return 0.

def test():
    alpha, beta = 1, 2
    N = 100
    X = np.random.normal(size = N)
    e = np.random.normal(size = N)
    Y = alpha + beta * X + e
    kernels = ['White','HansenHodrick','Bartlett','Parzen']
    for kern in kernels:
        theta, V, s, t = ols(Y, X, kern)
        print(kern, theta, s)
    
if __name__ == '__main__':
    print('Run test ...')
    test()