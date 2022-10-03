from Quadrature import *
from grid import *
from dirichlet_form import *
from utilities import *
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

##### Task 2.4 Grid refinement
n_disk = [11, 50, 99]
disks = [GetDisc(n) for n in n_disk]
fig, ax = plt.subplots(1, 3, figsize=(12, 4))


for i in range(3):
    ax[i].scatter(disks[i][0][:, 0], disks[i][0][:, 1])
    ax[i].set_title(f"N = {n_disk[i]}")

fig.show()

####2.5 
print("Task 2.5 tests:")

vec2 = lambda x,y : np.array([x, y])
p1 = vec2(0, 0)
p2 = vec2(1, 0)
p3 = vec2(0, 1)
test_f = lambda P: np.sin(2*np.pi*(P[0]**2 + P[1]**2))
print(quadrature2D(p1, p2, p3, 4, lambda P: 1))  # Tests for the constant function

A = get_stiffness_element(p1, p2, p3)
F = get_load_element(p1, p2, p3, 1, test_f)
print(A)
print(F)

for i in (3, 4):
    F = get_load_element(p1, p2, p3, i, test_f)
    print(F)

plt.figure(figsize = (4,3))
plt.title("A single element contribution")
im = plt.imshow(A)
plt.colorbar(im)
plt.show()
plt.close()


#Stiffness matrix
N = 101
n, e, b = GetDisc(N)

A = setup_system_matrix(n, e, b, return_entire_matrix=True)
A2 = setup_system_matrix(n, e, b, return_entire_matrix=False)
is_invertible = lambda A: (A.shape[0] == A.shape[1]) and (np.linalg.matrix_rank(A) == A.shape[0])
print(f"A invertible? {is_invertible(A)}")

fig, ax = plt.subplots(1,2,figsize = (12,6))
mx_plot = ax[0].imshow(A)
ax[0].set_title("The total stiffness matrix, singular")
plt.colorbar(mx_plot, ax = ax[0])


mx_plot2 = plt.imshow(A2)
ax[1].set_title("The internal stiffness matrix, non-singular")
plt.colorbar(mx_plot2, ax = ax[1])
plt.show()
plt.close()


#Task 2.8
print("Task 2.8 print-outs")
PI = np.pi
def f(P):
    x, y = P[0], P[1]
    ret = 0
    ret += -8 * PI * np.cos(2*PI*(x**2 + y**2))
    ret += 16 * PI**2 * (x**2 + y**2) * np.sin(2*PI*(x**2 + y**2))
    return ret


def fasit(P):
    return np.sin(2*PI*(P[0]**2 + P[1]**2))

N = 2000
u_fin, A, nodes, nrelm = solve_system(N, 4, f)
u_true = fasit(nodes.T)
f_visual = f(nodes.T)

fig, ax = plt.subplots(1,2,figsize = (10,4))
image0 = ax[0].tricontourf(nodes[:, 0], nodes[:, 1], u_fin, cmap = "inferno")
image2 = ax[1].tricontourf(nodes[:, 0], nodes[:, 1], u_true, cmap = "inferno")

plt.colorbar(image0, ax = ax[0])
plt.colorbar(image2, ax = ax[1])
ax[0].set_title("Numerical solution")
ax[1].set_title("Analytical solution")
#plt.savefig("dirichlet_solution.pdf")
plt.show()
plt.close()

ax = plt.subplot(projection='3d')
ax.set_xlabel("x-axis")
ax.set_ylabel("y-axis")
ax.plot_trisurf(nodes[:, 0], nodes[:, 1], u_fin-u_true, cmap = "inferno")
#plt.savefig("dirichlet_resid.pdf")
plt.show()
plt.close()

ax = plt.subplot()
#ax.set_title("Numerical-Analytic residual")
bilde = ax.tricontourf(nodes[:, 0], nodes[:, 1], u_fin-u_true, cmap = "inferno")
plt.colorbar(bilde)
ax.set_aspect('equal')
#plt.savefig("diri_cont.pdf")
plt.show()
plt.close()


convergence1()
