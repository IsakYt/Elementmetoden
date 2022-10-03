import numpy as np
from scipy.stats import linregress
from time import time
import matplotlib.pyplot as plt
from dirichlet_form import *
from neumann_form import *

def fasit(P):
    return np.sin(2*PI*(P[0]**2 + P[1]**2))

def g(P):
    x, y = P[0], P[1]
    return 4 * PI * np.sqrt(x**2 + y**2) * np.cos(2 * PI * (x**2 + y**2))

PI = np.pi
def f(P):
    x, y = P[0], P[1]
    ret = 0
    ret += -8 * PI * np.cos(2*PI*(x**2 + y**2))
    ret += 16 * PI**2 * (x**2 + y**2) * np.sin(2*PI*(x**2 + y**2))
    return ret

def get_error(uh, u, A):
    resid = u[0:len(A)] - uh[0:len(A)]
    norm = abs(resid.T @ A @ resid)**0.5
    return norm

def convergence1(save = False):
    
    #do not use num_nodes1 unless you have 5 minutes to spare
    #num_nodes1 = (10, 50, 100, 500, 1_000, 5_000, 10_000, 50_000)
    num_nodes = (10,50,100,500,1000,5000)
    
    elemes = []
    errors = []
    for N_nod in num_nodes:
        tid = time()
        Uh, Ah, nodes, nrelm = solve_system(N_nod,4,f)
        tid2 = time()
        print(f"{N_nod} Nodes took {tid2 - tid} seconds")
        elemes.append(nrelm)
        Bh = Ah.toarray()
        U = fasit(nodes.T)
        norm = get_error(Uh,U,Bh)
        errors.append(norm)
    
    plt.loglog(elemes,errors)
    plt.xlabel("Number of elements")
    plt.ylabel(r"Energy norm $u-u_h$")
    
    plt.grid()
    if save:
        plt.savefig("dirichlet_error.pdf")
        
    else:
        plt.title(r"Convergence rate given norm $\sqrt{\vec{v}^T A \vec{v}} $")
        
    plt.show()
    
def convergence2(save = False):
    #do not use num_nodes1 unless you have 5 minutes to spare
    #num_nodes1 = (10, 50, 100, 500, 1_000, 5_000, 10_000, 50_000)
    num_nodes = (10,50,100,500,1000,5000)
    
    elemes = []
    errors = []
    for N_nod in num_nodes:
        tid = time()
        Uh, Ah, nodes, nrelm, F = solve_neu_system(N_nod,4,f,g)
        tid2 = time()
        print(f"{N_nod} Nodes took {tid2 - tid} seconds")
        elemes.append(nrelm)
        Bh = Ah.toarray()
        U = fasit(nodes.T)
        norm = get_error(Uh,U,Bh)
        errors.append(norm)
    
    plt.loglog(elemes,errors)
    plt.xlabel("Number of elements")
    plt.ylabel(r"Energy norm $u-u_h$")
    
    plt.grid()
    if save:
        plt.savefig("dirichlet_error.pdf")
        
    else:
        plt.title(r"Convergence rate given norm $\sqrt{\vec{v}^T A \vec{v}} $")
        
    plt.show()
    

