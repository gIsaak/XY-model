import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
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
        self.nbr_hood = [((i // L) * L + (i + 1) % L, (i + L) % self.N,
            (i // L) * L + (i - 1) % L, (i - L) % self.N)
                                    for i in range(self.N)] #NSEW

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

    def Wolff(self):
        """Performs one Wolff's single custer algorithm iteration"""
        cluster = np.ones(self.state.shape, dtype=float)
        location = np.random.randint(0, self.N)
        cluster = self._grow_cluster(cluster, location)
        self.state *= cluster
        return self.state

    def Wolff_animation(self, times, delay=20000):
        """Animated Wolff's single custer algorithm
           In: times (int): number of repetition times
               delay (int): delay between frames in ms"""
        def update(i):
            image.set_array(self.Wolff())
            return image
            
        fig = plt.figure()
        ax = fig.add_subplot(111)
        image = ax.imshow(self.state, cmap=plt.cm.PiYG, origin='lower')
        ax.set_title('T = %.1f  J = %.1f' %(self.T, self.J))
        ani = animation.FuncAnimation(fig, update, frames=times, interval=delay, blit=True, save_count=1000)
        plt.show()


    def _get_ij(self, num):
        "Private method to obtain lattice indices i, j for spin number num"
        return num//self.L, num%self.L

    def plot_state(self):
        """Plots system state"""
        plt.matshow(self.state, cmap=plt.cm.PiYG)
#        plt.show()
        
    def plot_animation(self):
        """Plots system state as animation"""
        return plt.matshow(self.state, cmap=plt.cm.PiYG)



##### Isaccos way of animation #####
        
#L, J, T = 10, 1, 2
#system = Ising(L, J, T)
##this is with animation
#system.Wolff_animation(5000, delay=10000)
##this is without (one iteration)
##system.Wolff()

##### Ludwigs way of animation #####
        
# to stop interactive plotting
plt.ioff()
# initialisation of list to safe the state data _a: singele T animation _f: final states for various T
lattice_a = list()
lattice_f = list()
temps = [1.5,1.6,1.7,1.8,1.9,2,2.1,2.2,2.3,2.4,2.5,2.6]
for t in temps:
    L, J, T = 50, 1, t
    system = Ising(L, J, T)
    nsteps = 50
    for i in range(nsteps):
        system.Wolff()
        # next line saves the plots for singe temperature animation
#        lattice_a.append(system.plot_animation())
        plt.close("all")
        
    lattice_f.append(system.plot_animation())
    plt.close("all")


l = list()
for i in lattice_f:
    l.append(i._A)
    

##### Plotting #####
fig = plt.figure()
im = plt.imshow(l[0], interpolation="none", cmap="Blues")
title = plt.title("T = {}, J = {}".format(J,T))

def update(t):
    im.set_array(l[t])
    title.set_text("J = {}, T = {}, frame no.: {}".format(J,temps[t],str(t)))

ani = animation.FuncAnimation(fig, func=update, frames=nsteps, repeat=False, interval=1000)
plt.show()