import numpy as np
import sympy as sp
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from poisson import Poisson

from lagrangebasis import Lagrangebasis
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

x, y = sp.symbols("x,y")

# Below we create a solver that reuses some of the implementation from
# the 1D solver in poisson.py.


class Poisson2D:
    r"""Solve Poisson's equation in 2D::

        \nabla^2 u(x, y) = f(x, y), x, y in [0, L] x [0, L]

    with Dirichlet boundary conditions.
    """

    def __init__(self, L: float):
        self.p = Poisson(L)  # we can reuse some of the code from the 1D case

    def create_mesh(self, N: int) -> tuple[np.ndarray, np.ndarray]:
        """Return a 2D Cartesian mesh

        Parameters
        ----------
        N : int
            The number of uniform intervals in both x and y directions
        Returns
        -------
        xij : 2D array
            The x-coordinates of the mesh
        yij : 2D array
            The y-coordinates of the mesh
        """
        xi = self.p.create_mesh(N)
        xij, yij = np.meshgrid(xi, xi, indexing="ij", sparse=True)
        return xij, yij

    def laplace(self, N: int) -> sparse.lil_matrix:
        """Return a vectorized Laplace operator

        Parameters
        ----------
        N : int
            The number of uniform intervals in both x and y directions

        Returns
        -------
        A : scipy sparse LIL matrix
            The vectorized Laplace operator
        """
        D2 = self.p.D2(N, self.p.L/N).toarray()
        return (sparse.kron(D2, sparse.eye(N+1)) + sparse.kron(sparse.eye(N+1), D2))

    def assemble(
        self, N: int, f: sp.Expr, ue: sp.Expr
    ) -> tuple[sparse.csr_matrix, np.ndarray]:
        """Return assembled coefficient matrix A and right hand side vector b

        Parameters
        ----------
        Nx : int
            The number of uniform intervals in both x and y directions
        f : Sympy expression
            The right hand side as a Sympy expression in x and y
        ue : Sympy expression
            The exact solution as a Sympy expression in x and y

        Returns
        -------
        A : scipy sparse CSR matrix
            Coefficient matrix
        b : 1D array
            Right hand side vector

        Note
        ----
        Compute the Kronecker product of the 1D Laplace operator with itself
        to create the 2D Laplace operator. Then, assemble the right-hand side
        vector b by evaluating the function f at the mesh points and applying
        Dirichlet boundary conditions using the exact solution ue.

        """
        xij, yij = self.create_mesh(N)
        b = self.meshfunction(f, xij, yij).ravel()
        bnds = self.get_boundary_indices(N)
        b[bnds] = self.meshfunction(ue, xij, yij).ravel()[bnds]

        A = self.laplace(N).tolil()
        for i in bnds:
            A[i] = 0
            A[i, i] = 1

        return A.tocsr(), b

    def meshfunction(self, u: sp.Expr, xij: np.ndarray, yij: np.ndarray) -> np.ndarray:
        """Return Sympy function as mesh function

        Parameters
        ----------
        u : Sympy function

        Returns
        -------
        array - The input function as a mesh function
        """

        if isinstance(u, sp.core.numbers.Integer):
            N = len(xij.ravel())
            return sp.lambdify((x, y), u)(xij, yij)*np.ones((N, N))

        return sp.lambdify((x, y), u)(xij, yij)

    def get_boundary_indices(self, N: int) -> np.ndarray:
        """Return indices of vectorized matrix that belong to the boundary"""

        B = np.ones((N+1, N+1), dtype=bool)
        B[1:-1, 1:-1] = 0

        return np.where(B.ravel() == 1)[0]

    def l2_error(self, u: np.ndarray, ue: sp.Expr) -> float:
        """Return l2-error

        Parameters
        ----------
        u : array
            The numerical solution (mesh function)
        ue : Sympy expression
            The exact solution

        Returns
        -------
        float - The l2-error

        """
        N = u.shape[0] - 1
        xij, yij = self.create_mesh(N)
        return (1/N)*np.linalg.norm((u - self.meshfunction(ue, xij, yij)))

    def __call__(self, N: int, ue: sp.Expr) -> np.ndarray:
        """Solve Poisson's equation with a given manufactured solution

        Parameters
        ----------
        Nx : int
            The number of uniform intervals in both x and y directions
        ue : Sympy expression
            The exact solution

        Returns
        -------
        The solution as a Numpy array

        """
        A, b = self.assemble(N, sp.diff(ue, x, 2) + sp.diff(ue, y, 2), ue)
        return sparse_linalg.spsolve(A, b.ravel()).reshape((N + 1, N + 1))

    def convergence_rates(self, ue: sp.Expr, m: int = 6):
        E = []
        h = []
        N0 = 8
        for _ in range(m):
            u = self(N0, ue)
            E.append(self.l2_error(u, ue))
            h.append(self.p.L / N0)
            N0 *= 2
        r = [np.log(E[i - 1] / E[i]) / np.log(h[i - 1] / h[i]) for i in range(1, m, 1)]
        return r, np.array(E), np.array(h)

    def _lagrangefunction(self, U: np.ndarray, xval, yval):
        N = U.shape[0] - 1
        xij, yij = self.create_mesh(N)
        xij, yij = xij.ravel(), yij.ravel()

        # If right next to border, higher order interpolation
        # requires interpolating points too far away
        # from (xval, yval), which can increase inaccuracy,
        # so if next to border, do linear, otherwise, do cubic
        h = self.p.L/N
        if xval <= h or xval >= self.p.L - h or yval <= h or yval >= self.p.L - h:
            order = 1
        else:
            order = 3

        x_idx = round(xval/h) - (order)//2
        y_idx = round(yval/h) - (order)//2

        # We can still get issues with index error when being too
        # close to the border, so we double check the indices
        x_idx = min(N-order, max(0, x_idx))
        y_idx = min(N-order, max(0, y_idx))

        lx, ly = Lagrangebasis(xij[x_idx:x_idx+order+1], x), Lagrangebasis(yij[y_idx:y_idx+order+1], y)

        f = 0

        for i in range(order+1):
            for j in range(order+1):
                f += lx[i]*ly[j]*U[x_idx + i, y_idx + j]

        return sp.lambdify((x, y), f)

    def eval(self, U: np.ndarray, x: float, y: float) -> float:
        """Return u(x, y)

        Parameters
        ----------
        x, y : numbers
            The coordinates for evaluation

        Returns
        -------
        The value of u(x, y)

        """
        f = self._lagrangefunction(U, x, y)

        return f(x, y)

def test_create_mesh():
    sol = Poisson2D(1)
    xij_computed, yij_computed = sol.create_mesh(5)

    xij_expected = np.array([
        [0.0],
        [0.2],
        [0.4],
        [0.6],
        [0.8],
        [1.0]
    ])

    yij_expected = np.array([
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ])

    msg = f"create_mesh gave xij = {xij_computed} instead of expected {xij_expected}"
    assert np.sum(abs(xij_computed - xij_expected)) < 1e-12, msg
    msg = f"create_mesh gave yij = {yij_computed} instead of expected {yij_expected}"
    assert np.sum(abs(yij_computed - yij_expected)) < 1e-12, msg

def test_meshfunction():
    N = 5
    L = 1
    sol = Poisson2D(L)
    ue = sp.exp(sp.cos(4 * sp.pi * x) * sp.sin(2 * sp.pi * y))
    xij, yij = sol.create_mesh(N)
    u_computed = sol.meshfunction(ue, xij, yij)

    interval = np.linspace(0, L, N+1)
    X, Y = np.meshgrid(interval, interval, indexing="ij")
    u_expected = np.exp(np.cos(4*np.pi*X) * np.sin(2*np.pi*Y))

    msg = f"meshfunction gave this u\n{u_computed}\ninstead of expected\n{u_expected}"
    assert np.sum(abs(u_computed - u_expected)) < 1e-12, msg

def test_assemble():
    N = 5
    L = 1
    sol = Poisson2D(L)
    ue = sp.exp(sp.cos(4 * sp.pi * x) * sp.sin(2 * sp.pi * y))
    f = 12*x**2 + 6*y
    f = ue.diff(x, 2) + ue.diff(y, 2)
    xij, yij = sol.create_mesh(N)
    A_computed, b_computed = sol.assemble(N, f, ue)

    b_expected = sp.lambdify((x, y), ue)(xij, yij)
    b_expected[1:-1,1:-1] = sp.lambdify((x, y), f)(xij[1:-1, :], yij[:, 1:-1])
    b_expected = b_expected.ravel()

    msg = f"assemble gave this b\n{b_computed}\ninstead of expected\n{b_expected}"
    assert np.linalg.norm(b_computed - b_expected) < 1e-12, msg

def test_convergence_poisson2d():
    # This exact solution is NOT zero on the entire boundary
    ue = sp.exp(sp.cos(4 * sp.pi * x) * sp.sin(2 * sp.pi * y))
    sol = Poisson2D(1)
    r, _, _ = sol.convergence_rates(ue)
    assert abs(r[-1] - 2) < 1e-2


def test_interpolation():
    ue = sp.exp(sp.cos(4 * sp.pi * x) * sp.sin(2 * sp.pi * y))
    sol = Poisson2D(1)
    N = 100
    U = sol(N, ue)
    h = sol.p.L / N
    xij, yij = sol.create_mesh(N)
    U = sp.lambdify((x, y), ue)(xij, yij)
    assert abs(sol.eval(U, 0.52, 0.63) - ue.subs({x: 0.52, y: 0.63}).n()) < 1e-3
    assert abs(sol.eval(U, h / 2, 1 - h / 2) - ue.subs({x: h / 2, y: 1 - h / 2}).n()) < 1e-3

if __name__ == "__main__":
    test_create_mesh()
    test_meshfunction()
    test_assemble()
    test_convergence_poisson2d()
    test_interpolation()
    print("All tests passed!")
