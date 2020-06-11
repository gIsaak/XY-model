import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
sys.setrecursionlimit(10000)
np.random.seed(42)

class Ising:
    """2D Ising model with peridic boundary conditions
    Spins numbered sequentially from N-W to S-E from 1 to N-1"""
    def __init__(self, L, J, T):
        self.L = L
        self.N = L*L
        self.J = J
        self.T = T
        self.state = np.random.choice([-1.0, 1.0], size=(L,L))
        self.nbr_hood = [[(i - L) % self.N, (i + L) % self.N,
                    (i // L) * L + (i + 1) % L,
                    (i // L) * L + (i - 1) % L] for i in range(self.N)] #NSEW

    def __repr__(self):
        """Prints the state of the system as a string"""
        return str(self.state)

    def _grow_cluster(self, cluster, location):
        """Private method to recursively generate cluster for Wolff's single cluster algorithm
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
                if np.random.rand() < (1 - np.exp(-2*self.J/self.T)):
                    cluster = self._grow_cluster(cluster, nbr)
        return cluster

    def Wolff(self):
        """Performs one Wolff's single custer algorithm iteration"""
        location = np.random.randint(0, self.N+1)
        cluster = self._grow_cluster(np.ones(self.state.shape, dtype=float), location)
        self.state = np.multiply(self.state,cluster)
        return self.state

    def Wolff_animation(self, times, delay=20):
        """Animated Wolff's single custer algorithm
           In: times (int): number of repetition times
               delay (int): delay between frames in ms"""
        def update(i):
            image.set_array(self.Wolff())
            return image,
        fig = plt.figure()
        ax = fig.add_subplot(111)
        image = ax.imshow(self.state, cmap=plt.cm.PiYG, origin='lower')
        ax.set_title('T = %.1f  J = %.1f' %(self.T, self.J))
        ani = animation.FuncAnimation(fig, update, times, interval=delay, blit=True, save_count=1000, repeat=False)
        plt.show()

    def _get_ij(self, num):
        "Private method to obtain lattice indices i, j for spin number num"
        return num//self.L, num%self.L

    def plot_state(self):
        """Plots system state"""
        plt.matshow(self.state, cmap=plt.cm.PiYG)
        plt.show()

## There is something wrong with temperature
L, J, T = 100, 1, 1
ising = Ising(L, J, T)
ising.Wolff_animation(10000, delay=100)
