import numpy as np
import matplotlib.pyplot as plt
from sklearn import gaussian_process as gp

class VanillaGP():
    def __init__(self, data):
        self.kernel = gp.kernels.RBF(length_scale=1, length_scale_bounds=(1e-1,1e1))
        self.gp = gp.GaussianProcessRegressor(kernel=self.kernel,n_restarts_optimizer=50)
        self.x = data.x.reshape(-1,1)
        self.y = data.y.reshape(-1,1)

    def plot(self):
        self.gp.fit(self.x, self.y)
        _x = np.linspace(-5, 5, 10000)[:, None]
        _y, _s = self.gp.predict(_x, return_std=True)
        plt.figure(figsize=(10, 5))
        lw = 2
        plt.scatter(self.x, self.y, c='k', label='data')
        plt.plot(_x, np.sin(_x), color='navy', lw=lw, label='True')
        plt.plot(_x, _y, color='darkorange', lw=lw,
                 label='GPR (%s)' % self.gp.kernel_)
        _y = _y.reshape(-1)
        _s = _s.reshape(-1)
        plt.fill_between(_x[:, 0], _y - 2*_s, _y + 2*_s, color='orange', alpha=0.2)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim(-5, 5)
        plt.ylim(-4, 4)
        plt.legend(loc="best",  scatterpoints=1, prop={'size': 8})
        plt.show()

class Dataset():
    def __init__(self, N):
        min, max = -5, 5
        self.x = np.random.uniform(min, max, N)
        epsilon = np.random.normal(loc=0, scale=1, size=N)
        self.y = np.sin(self.x+epsilon)

data = Dataset(10)
gp = VanillaGP(data)
gp.plot()
