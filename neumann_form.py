import numpy as np
from Quadrature import *
from dirichlet_form import *
from grid import *

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
    """
    Builds the load vector for the mixed boundary condition case
    input:
        nodes: arr, xy coordinates of our nodes
        elements: arr, global indexing of the nodes
        boundary: arr, indices of the elements which lie on the boundary
        Nq: int, number of points used in quadrature
        f: function pointer, the load function of the BVP
        g: function pointer, the neumann boundary function.
    return:
        F: arr, the load vector
    """
    sys_size = len(nodes)
    F = np.zeros((sys_size,))
    
    #Contributions from internal nodes
    for el in elements: 
        #print(el)
        p1 = nodes[el[0]]
        p2 = nodes[el[1]]
        p3 = nodes[el[2]]

        Fk = get_load_element(p1, p2, p3, Nq, f)

        for a in range(3):
            F[el[a]] += Fk[a]

    #Contributiosn for boundary nodes
    for bd in boundary:

        #Check y-coordiante for neumann
        if (nodes[bd[0]][1] > 0):
            #Line integration happens now
            Gk = get_boundary_load(nodes[bd[0]],nodes[bd[1]], 4, g)

            for i in range(2):
                F[bd[i]] += Gk[i]


        #Dirichlet nodes
        else:
            for i in range(2):
                F[bd[i]] = 0
            

    #bdry_nodes = set(list(boundary.flatten())) 
    #for node in bdry_nodes:
    #    F[node] = 0
    
    return F

def setup_system_matrix_neumann(nodes, elements, boundary):
    """
    Builds the system amtrix for our mixed boundary condition case
    input:
        nodes: arr, xy coordinates of our nodes
        elements: arr, global indexing of the nodes
        boundary: arr, indices of the elements which lie on the boundary
    return:
        A: arr, the stiffness matrix.
        
    """
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
    
    #Handling the dirichelt half of the boundary condtion
    bdry_nodes = set(list(boundary.flatten())) 
    for node in bdry_nodes:
        #If negative y-coordinate
        if nodes[node][1] < 0:
            A[:, node] = 0
            A[node, :] = 0
            A[node, node] = 1
        else:
            pass
    
    return A

def solve_neu_system(N_nod, Nq, f, g):
    """
    Solves the equation Au = F for the mixed boundary condition case
    input:
        N_nod: int, number of nodes used
        Nq: int, number of points used in quadrature
        f: function pointer, the load function f
        g: function pointer, the neumann boundary function
    return:
        u_fin: arr, the numerical solution
        A: arr, the stiffness matrix
        n: arr, the xy coordinates of the nodes
        E: int, number of elements
        F: arr, the load vector
    """
    n, e, b = GetDisc(N_nod)
    F = setup_load_vector_neumann(n, e, b, Nq, f,g)
    A = sp.csr_matrix(setup_system_matrix_neumann(n, e, b))

    u_fin = np.zeros(N_nod, dtype = float)

    u, info = sp.linalg.cg(A,F)  # Conjugate Gradient to quickly solve the system
    u_fin[0:len(u)] = u
    E = len(e)

    return u_fin.T, A, n, E, F