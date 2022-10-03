import numpy as np
from grid import *
from Quadrature import *
import scipy.sparse as sp

def get_stiffness_element(p1, p2, p3):
    """
    Creates elemental contrib. to stiffness matrix from the given coordinates.
    Parameters:
        p1, p2, p3: Corner points of triangular element, 2D-vectors
    Returns:
        A * area: elemental contribution
    """
    ### Implementing the contribution:
    ### (A^k)_a,b = (area)^k * ((c_x,a) * (c_x,b) + (c_y,a) * (c_y,b))
    A = np.zeros((3,3))

    coord_mx = np.array([[1, p1[0], p1[1]], [1, p2[0], p2[1]], [1, p3[0], p3[1]]])
    # Coefficients of 2D linear polynomials: p(x, y) = c + c_x * x + c_y * y
    linear_coeffs = np.linalg.inv(coord_mx)

    # Calculating the area of the inputted triangle to scale the contribution correctly
    v1 = p3 - p1
    v2 = p3 - p2
    area = 0.5 * (v1[0] * v2[1] - v1[1] * v2[0])
    
    #Here the elemental contribution is simply the integral over constant functions,
    #as we are "integrating" over the derivatives of linear polynomials, so in this case no quadrature is needed
    for i in range(3):
        for j in range(3):
            # Row 0 is constant offsets, row 1 is x-coefficients, row 2 is y-coefficients
            A[i, j] = linear_coeffs[1, i] * linear_coeffs[1, j] + linear_coeffs[2, i] * linear_coeffs[2, j]
    
    return A * area  # Returning correctly scaled elemental contribution

def get_load_element(p1, p2, p3, Nq, f):
    """
    Creates elemental contrib. to load vector from given coordinates. 
    Integrates load function with linear elements using Gauss quadrature.
    Parameters:
        p1, p2, p3: Corner points of triangular element, 2D-vectors.
        Nq: Number of points used in the gauss quadrature
        f: Load function
    Returns:
        Fk: Elemental contribution to load vector
    """
    ### Implementing the contribution:
    ### (F^k)_a = integral[ f(x, y) * H_a(x, y) dxdy]
    Fk = np.zeros((3,))
    coord_mx = np.array([[1, p1[0], p1[1]], [1, p2[0], p2[1]], [1, p3[0], p3[1]]])
    # Coefficients of 2D linear polynomials: p(x, y) = c + c_x * x + c_y * y
    linear_coeffs = np.linalg.inv(coord_mx)

    for i in range(3):
        # H_a(x, y) = c_a + c_x,a * x + c_y,a * y
        H = lambda P: linear_coeffs[0, i] + linear_coeffs[1, i] * P[0] + linear_coeffs[2, i] * P[1]
        Fk[i] = quadrature2D(p1, p2, p3, Nq, lambda P: f(P) * H(P))  # Integration routine for suitable function
    
    return Fk

def setup_system_matrix(nodes, elements, boundary, return_entire_matrix=False):
    """
    input:
        nodes: arr, xy coordinates of our nodes
        elements: arr, global indexing of the nodes
        boundary: arr, indices of the elements which lie on the boundary
        return_entire_matrix: bool, if True returns overdependent, singualar, system
    return:
        A: arr, the internal stiffness matrix
        B: arr, the total stiffenss matrix
    """
    sys_size = len(nodes)
    # Initializing system matrix for speed, could consider to directly set up a sparse matrix
    A = np.zeros((sys_size, sys_size))
    
    for el in elements:
        # Corner points of element "el"
        # Array indexing performs the role of local-to-global map
        p1 = nodes[el[0]]
        p2 = nodes[el[1]]
        p3 = nodes[el[2]]

        # The element's contribution matrix
        Ak = get_stiffness_element(p1, p2, p3)

        # Updating the corresponding locations in the system matrix
        for a in range(3):
            for b in range(3):
                A[el[a], el[b]] += Ak[a,b]

    # Homogeneous Dirichlet BC's mean that nodal values on the boundary don't
    # need computing. Since they're the last nodes in the GetDisc-mesh, we can
    # treat them by returning a submatrix, effectively never solving for the known values of zero.
    not_needed = len(boundary)
    B = A[0:sys_size-not_needed,0:sys_size-not_needed]

    #bdry_nodes = set(list(boundary.flatten()))  # All global node numbers (only once) that are on the boundary
    #for node in bdry_nodes:
    #    A[:, node] = 0
    #    A[node, :] = 0
    #    A[node, node] = 1

    if return_entire_matrix:
        return A
    else:
        return B
    
    
def setup_load_vector(nodes, elements, boundary, Nq, f):
    sys_size = len(nodes)
    # Initializing for speed
    F = np.zeros((sys_size,))

    for el in elements: 
        # Local-to-global map from array indexing
        p1 = nodes[el[0]]
        p2 = nodes[el[1]]
        p3 = nodes[el[2]]
        # Elemental contribution to load vector
        Fk = get_load_element(p1, p2, p3, Nq, f)
        # Updating the corresponding locations based on the local-to-global map
        for a in range(3):
            F[el[a]] += Fk[a]

    # Homogeneous Dirichlet BC's mean that nodal values on the boundary don't
    # need computing. Since they're the last nodes in the GetDisc-mesh, we can
    # treat them by returning a subvector, effectively never solving for the known values of zero.
    not_needed = len(boundary)
    G = F[0:sys_size-not_needed]

    #bdry_nodes = set(list(boundary.flatten()))  # All global node numbers (only once) that are on the boundary
    #for node in bdry_nodes:
    #    F[node] = 0
    
    return G

def solve_system(N_nod, Nq, f):
    """
    Builds and solves the system Au = F for the poisson problem, over the unit disk
    input:
        N_nod: int, number of nodes in the system
        Nq: int, number of quadrature points
        f: function pointer, the load function f
    """
    n, e, b = GetDisc(N_nod)
    F = setup_load_vector(n, e, b, Nq, f)
    A = sp.csr_matrix(setup_system_matrix(n, e, b))

    u_fin = np.zeros(N_nod, dtype = float)

    #u = sp.linalg.spsolve(A, F)
    u, info = sp.linalg.cg(A,F)  # Conjugate Gradient to quickly solve the system
    u_fin[0:len(u)] = u
    E = len(e)
    return u_fin.T, A, n, E