import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
sys.setrecursionlimit(10000)
np.random.seed(42)

def pol2cart(phi, r):
    """Gets Cartesian coordinates from polar ones"""
    x = np.multiply(r, np.cos(phi))
    y = np.multiply(r, np.sin(phi))
    return x, y

def cart2pol(x, y):
    """Gets polar coordinates from cartesian ones"""
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return phi, r

def get_comp(phi1, r1, phi2, r2):
    """Decomposes vector (phi1, r1) along specified vector (phi2, r2)
        Input in polar coordinates
        Output cartesian coordinates
        (x_p, y_p): parallel component
        (x_s, y_s): perpendicular component"""
    x1, y1 = pol2cart(phi1, r1)
    x2, y2 = pol2cart(phi2, r2)
    dot = np.multiply(x1, x2) + np.multiply(y1, y2)
    x_p = np.multiply(np.divide(dot, r2**2), x2)
    y_p = np.multiply(np.divide(dot, r2**2), y2)
    x_s = x1 - x_p
    y_s = y1 - y_p
    return x_p, y_p, x_s, y_s

class XY:
    """2D XY model with peridic boundary conditions
        Spin expressed with its angle with respect to x axis in [-pi, pi)
        Spins numbered sequentially from N-W to S-E from 1 to N-1"""
    def __init__(self, L, J, T):
        self.L = L
        self.N = L*L
        self.J = J
        self.T = T
        self.state = np.random.uniform(-np.pi, np.pi, size=(L,L))
        self.eps = None
        self.x_p = None #parallel
        self.y_p = None
        self.r_p = None
        self.x_s = None #senkrecht
        self.y_s = None
        self.nbr_hood = [[(i - L) % self.N, (i + L) % self.N,
                    (i // L) * L + (i + 1) % L,
                    (i // L) * L + (i - 1) % L] for i in range(self.N)] #NSEW

    def _project(self, u):
        """Private method to project lattice of spins onto u
            u in interval [-pi, pi)
            In: u (float): direction u (angle)"""
        u_x, u_y = pol2cart(u, 1)
        self.x_p, self.y_p, self.x_s, self.y_s = get_comp(self.state, 1, u, 1)
        self.r_p = np.sqrt(self.x_p**2 + self.y_p**2)
        self.eps = np.sign(self.x_p/u_x + self.y_p/u_y)

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
            if cluster[i_nbr, j_nbr] == 1 and self.eps[i_nbr, j_nbr] == self.eps[i, j]:
                J = self.J*self.r_p[i,j]*self.r_p[i_nbr, j_nbr]
                if np.random.rand() < (1 - np.exp(-2*J/self.T)):
                    cluster = self._grow_cluster(cluster, nbr)
        return cluster

    def Wolff(self):
        """Performs one Wolff's single custer algorithm iteration"""
        # embed Ising onto XY
        u = np.random.uniform(-np.pi, np.pi)
        self._project(u)
        # do wolff on embedded Ising
        location = np.random.randint(0, self.N)
        cluster = self._grow_cluster(np.ones(self.eps.shape, dtype=float), location)
        self.eps = np.multiply(self.eps, cluster)
        # compute new XY lattice state
        self.x_p = np.multiply(cluster, self.x_p)
        self.y_p = np.multiply(cluster, self.y_p)
        self.state, r = cart2pol(self.x_p + self.x_s, self.y_p + self.y_s)
        return self.state

    def Wolff_animation(self, times, delay=20):
        """Animated Wolff's single custer algorithm
           In: times (int): number of repetition times
               delay (int): delay between frames in ms"""
        def update(i):
            self.Wolff()
            X, Y = pol2cart(self.state,1)
            X = np.asarray(X)
            Y = np.asarray(Y)
            x = np.arange(0, self.L, 1)
            y = np.arange(0, self.L, 1)
            q = ax.quiver(x, y, X, Y, pivot='mid', scale_units='xy')
            return q,
        fig, ax = plt.subplots()
        ax.set_title('T = %.1f  J = %.1f' %(self.T, self.J), fontsize=20)
        ani = animation.FuncAnimation(fig, update, times, interval=delay,
                                    blit=True, save_count=1000, repeat=False)
        plt.show()

    def _get_ij(self, num):
        "Private method to obtain lattice indices i, j for spin number num"
        return num//self.L, num%self.L

    def __repr__(self):
        """Prints the state of the system as a string"""
        return str(self.state)

    def plot_state(self):
        """Plots system state"""
        plt.matshow(self.state, cmap=plt.cm.twilight)
        plt.colorbar()
        plt.show()


## There is something wrong with temperature
L, J, T = 10, 1, 1.2
xy = XY(L, J, T)
xy.Wolff_animation(100, delay=1000)
