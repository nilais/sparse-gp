import numpy as np
import GPy as gp
import matplotlib as mp
import matplotlib.pyplot as pl

class GaussianProcess():

    def __init__(self, kernel=None):
        if kernel is None:
            kernel = gp.kern.RBF(1)
        self.kernel = kernel

    def fit(self, data):
        n = data.n
        X, y = data.x, data.y
        self.gp = gp.models.GPRegression(X,y)
        self.gp.optimize('bfgs')

    def plot(self):
        self.gp.plot()
        mp.pylab.show(block=True)



class Dataset():

    def __init__(self, N):
        noise_var = 0.1
        k = gp.kern.RBF(1)
        self.n = N
        self.x = np.linspace(0,10,N)[:,None]
        self.y = np.random.multivariate_normal(np.zeros(N),k.K(self.x)+np.eye(N)*np.sqrt(noise_var)).reshape(-1,1)

d = Dataset(50)
g = GaussianProcess()
g.fit(d)
g.plot()
