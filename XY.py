import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
sys.setrecursionlimit(10000)
np.random.seed(42)

def sawtooth(x, T=2*np.pi, A=np.pi):
    """Sawtooth function of period T and amplitude A centered in 0
        In: x (nparray): sawtooth input
            T (float): period
            A (float): amplitude"""
    return 2*A*(x/T - np.floor(0.5 + x/T))

def addPolar(phi1, r1, phi2, r2):
    """Add two (array of) vectors r1 + r2 in polar coordinates
       angle from [-pi, pi)
        In: phi1 (nparray): angle of first vector
            r1 (nparray): modulus of first vectors
            phi2 (nparray): angle of second vector
            r2 (nparray): modulus of second vectors
            T (float): period
        Out: r (nparray): modulus of result
             phi (ndarray): angle of result"""
    cos = np.cos(phi2 - phi1)
    sin = np.sin(phi2 - phi1)
    r1r2 = np.multiply(r1, r2)
    r = np.sqrt(r1**2 + r2**2 + 2*np.multiply(r1r2, cos))
    phi = phi1 + np.arctan2(np.multiply(r2, sin), r1 + np.multiply(r2, cos))
    phi = sawtooth(phi)
    return phi, r

def subtractPolar(phi1, r1, phi2, r2):
    """Subtract two (array of) vectors r1 + r2 in polar coordinates
       angle from [-pi, pi)
        In: phi1 (nparray): angle of first vector
            r1 (nparray): modulus of first vectors
            phi2 (nparray): angle of second vector
            r2 (nparray): modulus of second vectors
            T (float): period
        Out: r (nparray): modulus of result
             phi (ndarray): angle of result"""
    cos = np.cos(phi2 - phi1)
    sin = np.sin(phi2 - phi1)
    r1r2 = np.multiply(r1, r2)
    r = np.sqrt(r1**2 + r2**2 - 2*np.multiply(r1r2, cos))
    phi = phi1 + np.arctan2(-np.multiply(r2, sin), r1 - np.multiply(r2, cos))
    phi = sawtooth(phi)
    return phi, r

class XY:
    """2D XY model with peridic boundary conditions
        Spin expressed with its angle with respect to x axis in [-pi, pi)
        Spins numbered sequentially from N-W to S-E from 1 to N-1"""
    def __init__(self, L, J, T):
        self.L = L
        self.N = L*L
        self.J = J
        self.T = T
        self.state = np.random.uniform(-1, 1, size=(L,L))*np.pi
        self.eps = np.zeros(self.state.shape)
        self.r_p = np.zeros(self.state.shape) #parallel
        self.phi_p = np.zeros(self.state.shape)
        self.r_s = np.zeros(self.state.shape) #senkrecht
        self.phi_s = np.zeros(self.state.shape)
        self.nbr_hood = [[(i - L) % self.N, (i + L) % self.N,
                    (i // L) * L + (i + 1) % L,
                    (i // L) * L + (i - 1) % L] for i in range(self.N)] #NSEW

    def _project(self, u):
        """Private method to project lattice of spins onto u
            u in interval [-pi, pi)
            In: u (float): direction u (angle)"""
        cos = np.cos(self.state - u)
        self.eps = np.sign(cos)
        self.phi_p = sawtooth(np.multiply(self.eps, u))
        self.r_p = np.abs(cos)
        self.phi_s, self.r_s = subtractPolar(self.state, 1, self.phi_p, self.r_p)

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
                p = np.random.rand()
                if p < (1 - np.exp(-2*J/self.T)):
                    cluster = self._grow_cluster(cluster, nbr)
        return cluster

    def Wolff(self):
        """Performs one Wolff's single custer algorithm iteration"""
        # embed Ising onto XY
        u = np.random.uniform(-1, 1)*np.pi
        self._project(u)
        # do wolff on embedded Ising
        location = np.random.randint(0, self.N)
        cluster = self._grow_cluster(np.ones(self.eps.shape, dtype=float), location)
        self.eps = np.multiply(self.eps, cluster)
        # compute new XY lattice state
        self.phi_p = sawtooth(np.multiply(self.eps, self.phi_p))
        self.state, _ = addPolar(self.phi_s, self.r_s, self.phi_p, self.r_p)
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
        image = ax.imshow(self.state, cmap=plt.cm.twilight, origin='lower')
        ax.set_title('T = %.1f  J = %.1f' %(self.T, self.J))
        fig.colorbar(image)
        ani = animation.FuncAnimation(fig, update, times, interval=delay, blit=True, save_count=1000, repeat=False)
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
L, J, T = 10, 1, 3
xy = XY(L, J, T)
xy.Wolff_animation(10, delay=1000)
