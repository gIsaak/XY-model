import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
sys.setrecursionlimit(10000)
np.random.seed(42)

def dotUnit(phi1, phi2):
    """Dot product between two unit vectors in polar coordinates"""
    x1, y1 = pol2cart(phi1, 1)
    x2, y2 = pol2cart(phi2, 1)
    dot = np.multiply(x1, x2) + np.multiply(y1, y2)
    return dot

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
            x = np.arange(0, self.L, 1)
            y = np.arange(0, self.L, 1)
            q = ax.quiver(x, y, X, Y, pivot='mid')
            return q,
        fig, ax = plt.subplots(figsize = (8, 8))
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
        X, Y = pol2cart(self.state, 1)
        x = np.arange(0, self.L, 1)
        y = np.arange(0, self.L, 1)
        fig, ax = plt.subplots(figsize = (8, 8))
        ax.set_title('T = %.1f  J = %.1f' %(self.T, self.J), fontsize=20)
        q = ax.quiver(x, y, X, Y, pivot='mid')
        plt.show()

    def get_magnetization(self):
        """Computes the magnitude of magnetization of current configuration"""
        X, Y = pol2cart(self.state,1)
        return np.sqrt(np.mean(X)**2+np.mean(Y)**2)

    def get_energy(self):
        """Computes energy of current configuration"""
        E = 0
        for s in range(self.N):
            i, j = self._get_ij(s)
            s_ij = self.state[i,j]
            for nbr in self.nbr_hood[s]:
                i_nbr, j_nbr = self._get_ij(nbr)
                s_nbr = self.state[i_nbr, j_nbr]
                E -= self.J*np.cos(s_nbr-s_ij)
        return self.J*E/2

    def waste_time(self, steps):
        """Waste steps steps number of steps"""
        for i in range(steps):
            self.Wolff()

    def reset_state(self):
        """Resets system to original states for given L, J, T"""
        self.__init__(self.L, self.J, self.T)



### SIMULATION ###
L, J, T = 20, 1, 1
xy = XY(L, J, T)
# Initialize observables
steps = 5000
eq_steps = 2000
waste_steps = 3
T = np.linspace(0.5, 1.5, 20)

E_of_t = []
E2_of_t = []
M_of_t = []
M2_of_t = []
for t in T:
    print('Temperature: ', t)
    xy.T = t
    xy.reset_state()
    xy.waste_time(eq_steps)
    E_acc, E2_acc = [], []
    M_acc, M2_acc = [], []
    for i in range(eq_steps):
        xy.waste_time(waste_steps)
        # energy
        E = xy.get_energy()
        E_acc.append(E)
        E2_acc.append(E*E)
        # magnetization
        M = xy.get_magnetization()
        M_acc.append(M)
        M2_acc.append(M*M)
    E_of_t.append(np.mean(np.asarray(E_acc)))
    E2_of_t.append(np.mean(np.asarray(E2_acc)))
    M_of_t.append(np.mean(np.asarray(M_acc)))
    M2_of_t.append(np.mean(np.asarray(M2_acc)))
cv = (np.asarray(E2_of_t) - np.asarray(E_of_t)**2)/xy.T**2/xy.N
chi = (np.asarray(M2_of_t) - np.asarray(M_of_t)**2)*xy.N/xy.T

# plotting
# E
fig, ax1 = plt.subplots()
ax1.set_xlabel('T')
ax1.set_ylabel('E', color='b')
ax1.plot(T, E_of_t, color='b')
ax1.tick_params(axis='y')
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('Cv', color='g')  # we already handled the x-label with ax1
ax2.plot(T, cv, color='g')
ax2.tick_params(axis='y')
fig.tight_layout()  # otherwise the right y-label is slightly clipped


# M
fig, ax1 = plt.subplots()
ax1.set_xlabel('T')
ax1.set_ylabel('M', color='m')
ax1.plot(T, M_of_t, color='m')
ax1.tick_params(axis='y')
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('$\chi$', color='c')  # we already handled the x-label with ax1
ax2.plot(T, chi, color='c')
ax2.tick_params(axis='y')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

plt.show()
