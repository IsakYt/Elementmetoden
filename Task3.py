from Quadrature import *
from grid import *
from dirichlet_form import *
from utilities import *
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def g(P):
    x, y = P[0], P[1]
    return 4 * PI * np.sqrt(x**2 + y**2) * np.cos(2 * PI * (x**2 + y**2))

def get_boundary_load(p1, p2, Nq, g):
    """
    Creates elemental contrib. to load vector from given coordinates. 
    Integrates load function with linear elements using Gauss quadrature.
    Parameters:
        p1, p2: boundary nodes, 2D-vectors.
        Nq: Number of points used in the gauss quadrature
        g: boundary function
    Returns:
        Fk: Elemental contribution to load vector
    """
    Fk = np.zeros((2,))
    p_matrix = np.array([[p1[0],p1[1]],[p2[0],p2[1]]])
    CM = np.linalg.inv(p_matrix)
    
    for i in range(2):
        H = lambda P: CM[0,i]*P[0] + CM[1,i]*P[1]
        Fk[i] = quadrature1D(p1, p2, Nq, lambda P: g(P) * H(P))
        
    
    return Fk

def setup_load_vector_neumann(nodes, elements, boundary, Nq, f, g):
    sys_size = len(nodes)
    F = np.zeros((sys_size,))

    for el in elements: 
        #print(el)
        p1 = nodes[el[0]]
        p2 = nodes[el[1]]
        p3 = nodes[el[2]]

        Fk = get_load_element(p1, p2, p3, Nq, f)

        for a in range(3):
            F[el[a]] += Fk[a]

    ### add something with the boundary
    for bd in boundary:

        #Neumann
        if (nodes[bd[0]][1] > 0): #Må stille krav til y-koordinat
            Gk = get_boundary_load(nodes[bd[0]],nodes[bd[1]], 4, g)

            for i in range(2):
                F[bd[i]] += Gk[i]


        #Dirichlet
        else:
            for i in range(2):
                F[bd[i]] = 0
            

    #bdry_nodes = set(list(boundary.flatten())) 
    #for node in bdry_nodes:
    #    F[node] = 0
    
    return F

def setup_system_matrix_neumann(nodes, elements, boundary):
    sys_size = len(nodes)
    A = np.zeros((sys_size, sys_size))
    
    for el in elements:
        #Corner points of element number "el"
        p1 = nodes[el[0]]
        p2 = nodes[el[1]]
        p3 = nodes[el[2]]

        #The element's contribution matrix
        Ak = get_stiffness_element(p1, p2, p3)

        #Updating the corresponding locations in the system matrix
        for a in range(3):
            for b in range(3):
                A[el[a], el[b]] += Ak[a,b]

    bdry_nodes = set(list(boundary.flatten())) #sikker på at flatten funker her?
    for node in bdry_nodes:
        if nodes[node][1] < 0:
            A[:, node] = 0
            A[node, :] = 0
            A[node, node] = 1
        else:
            pass
    
    return A

def solve_neu_system(N_nod, Nq, f,g):
    n, e, b = GetDisc(N_nod)
    F = setup_load_vector_neumann(n, e, b, Nq, f,g)
    A = sp.csr_matrix(setup_system_matrix_neumann(n, e, b))

    u_fin = np.zeros(N_nod, dtype = float)

    u, info = sp.linalg.cg(A,F)  # Conjugate Gradient to quickly solve the system
    u_fin[0:len(u)] = u

    return u_fin.T, A, n, len(e), F



N_sys = 2000

_u, A, n, n_elm, F = solve_neu_system(N_sys, 4, f,g)
_F = np.zeros((N_sys,))
_F[0:F.shape[0]] = F

fig, ax = plt.subplots(1,2,figsize = (10,4))

im1 = ax[0].tricontourf(n[:, 0], n[:, 1], _F, cmap="inferno") 
im2 = ax[1].tricontourf(n[:, 0], n[:, 1], _u, cmap="inferno")
plt.colorbar(im1, ax = ax[0])
plt.colorbar(im2, ax = ax[1])
#plt.savefig("Mixed_load_and_sol.pdf")
plt.show()

u_true_2 = fasit(n.T)


ax = plt.subplot(projection='3d')
#ax.set_title("Numerical-Analytic residual")
ax.set_xlabel("x-axis")
ax.set_ylabel("y-axis")
ax.plot_trisurf(n[:, 0], n[:, 1], (_u-u_true_2), cmap = "inferno")
#plt.savefig("Mixed_residual.pdf")
plt.show()

avicii = np.linspace(-0.5,0.07,21)
im3 = plt.tricontourf(n[:,0],n[:,1],_u-u_true_2, avicii, cmap = "inferno")
plt.colorbar(im3)
#plt.savefig("Mixed_cont.pdf")
plt.show()

convergence2()