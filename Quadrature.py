import numpy as np

GUASS_WEIGHTS_1D = {1: [2], 
2: [1,1], 
3: [5/9, 8/9, 5/9], 
4: [(18-30**0.5)/36, (18+30**0.5)/36, (18+30**0.5)/36, (18-30**0.5)/36]}

GAUSS_NODES_1D = {1: [0],
2: [-1/3**0.5, 1/3**0.5],
3: [-(3/5)**0.5, 0, (3/5)**0.5],
4: [-((3+2*(6/5)**0.5)/7)**0.5, -((3-2*(6/5)**0.5)/7)**0.5, ((3-2*(6/5)**0.5)/7)**0.5, ((3+2*(6/5)**0.5)/7)**0.5]}

GAUSS_NODES_2D = {
    1: [[1/3, 1/3, 1/3]],
    3: [[1/2, 1/2, 0], [1/2, 0, 1/2], [0, 1/2, 1/2]],
    4: [[1/3, 1/3, 1/3], [3/5, 1/5, 1/5], [1/5, 3/5, 1/5], [1/5, 1/5, 3/5]]
}

GUASS_WEIGHTS_2D = {
    1: [1],
    3: [1/3, 1/3, 1/3],
    4: [-9/16, 25/48, 25/48, 25/48]
}


def quadrature1D(a,b,Nq,g):
    """
    1-dimensional guassian quadrature
    change of interval: https://en.wikipedia.org/wiki/Gaussian_quadrature#Change_of_interval 

    Parameters:
    a: float, start of integration domain
    b: float, end of integration domain
    Nq: int, number of evaluation points
    g: function pointer, function which is being integrated. g: R -> R
    
    Return:
    I: float, result of integration
    """


    P = lambda t: (1-t)*a/2 + (1+t)*b/2

    I = 0 
    c = np.linalg.norm(b-a,2)/2 

    for i in range(len(GAUSS_NODES_1D[Nq])):
        #function evaluation
        evl = g(P(GAUSS_NODES_1D[Nq][i])) 

        #updating integral sum
        I += GUASS_WEIGHTS_1D[Nq][i]*evl

    #Scaling the sum
    I = c*I
    return I

def quadrature2D(p1, p2, p3, Nq, g):
    """2D gaussian quadrature
    Parameters:
    p1, p2, p3: physical coordinates of the triangle corner points as vec3
    Nq: number of integration points; 1, 3 or 4
    g: function pointer to integrand

    Return:
    I: float, result of integration
    """

    nodes = GAUSS_NODES_2D[Nq]
    weights = GUASS_WEIGHTS_2D[Nq]
    v1 = p3 - p1
    v2 = p3 - p2
    area = abs(0.5 * (v1[0] * v2[1] - v1[1] * v2[0]))   

    I = 0
    for i in range(Nq):
        P = p3 + nodes[i][0] * (p1 - p3) + nodes[i][1] * (p2 - p3)
        I += weights[i] * g(P)
    
    return I*area