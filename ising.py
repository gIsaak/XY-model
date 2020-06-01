import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import time
np.random.seed(42)

class Ising:
    """2D Ising model with peridic boundary conditions
    Spins numbsered sequentially from N-W to S-E from 1 to N-1"""
    def __init__(self, L, J, T):
        self.L = L
        self.N = L*L
        self.J = J
        self.T = T
        self.state = np.random.choice([-1.0, 1.0], size=(L,L))
        self.nbr_hood = [[(i - L) % L*L, (i + L) % L*L,
                    (i // L) * L + (i + 1) % L,
                    (i // L) * L + (i - 1) % L] for i in range(L*L)] #NSEW

    def __repr__(self):
        """Prints the state of the system as a string"""
        return str(self.state)

    def _grow_cluster(self, cluster, location):
        """Private metod to recursively generate cluster for Wolff's single cluster algorithm
           In: cluster (nparray state.shape): initial cluster
                                                -1 = in the cluster
                                                 1 = not in the cluster
               location (int): spin number to be tried
           Out: cluster (nparray state.shape): fully grown cluster"""
        i, j = self._get_ij(location)
        cluster[i,j] = -1
        for nbr in self.nbr_hood[location]:
            i_nbr, j_nbr = self._get_ij(nbr)
            if cluster[i_nbr, j_nbr] == 1 and self.state[i_nbr, j_nbr] == self.state[i, j]:
                if np.random.rand() < 1 - np.exp(-2*self.J/self.T):
                    self._grow_cluster(cluster, nbr)
        return cluster

    def Wolff(self, show_cluster=False):
        """Performs one Wolff's single custer algorithm iteration"""
        cluster = np.ones(self.state.shape, dtype=float)
        location = np.random.randint(0, self.N)
        cluster = self._grow_cluster(cluster, location)
        self.state *= cluster
        if show_cluster == True:
            print((1-cluster)/2)


    def _get_ij(self, num):
        "Private method to obtain lattice indices i, j for spin number num"
        return num//self.L, num%self.L

    def plot_state(self):
        """Plots system state"""
        plt.matshow(self.state, cmap=plt.cm.PiYG)
        plt.show()


L, J, T = 100, 1, 1
system = Ising(L, J, T)
system.plot_state()
for i in range(5000):
    system.Wolff()
system.plot_state()
