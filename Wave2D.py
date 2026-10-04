import numpy as np
import sympy as sp
from scipy import sparse

x, y, t = sp.symbols("x,y,t")


class Wave2D:
    """Class for solving the 2D wave equation"""


    def create_mesh(
        self, N: int, sparse: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return 2D mesh created using np.meshgrid

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        sparse : bool, optional
            Whether to create a sparse mesh or not. Default is False.
        Returns
        -------
        xij : 2D array
            The x-coordinates of the mesh
        yij : 2D array
            The y-coordinates of the mesh"""
        xi = np.linspace(0, self.L, N + 1)
        xij, yij = np.meshgrid(xi, xi, indexing="ij", sparse=True)
        return xij, yij

    def D2(self, N: int) -> sparse.lil_matrix:
        """Return second order differentiation matrix

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Returns
        -------
        D : scipy sparse LIL matrix
            The second order differentiation matrix
        """
        D2 = sparse.diags([1., -2., 1.], [-1, 0, 1], (N + 1, N + 1), format="lil")
        D2[0, :4] = 2, -5, 4, -1
        D2[-1, -4:] = -1, 4, -5, 2
        return D2

    @property
    def w(self):
        """Return the dispersion coefficient"""
        return self.c

    def ue(self, mx: int, my: int) -> sp.Expr:
        """Return the exact standing wave

        Parameters
        ----------
        mx, my : int
            Parameters for the standing wave
        Returns
        -------
        ue : Sympy expression
            The exact solution as a Sympy expression in x, y and t
        """
        return sp.sin(mx * sp.pi * x) * sp.sin(my * sp.pi * y) * sp.cos(self.w * t)

    def meshfunction(self, u: sp.Expr, xij: np.ndarray, yij: np.ndarray, t0: float) -> np.ndarray:
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
            return sp.lambdify((x, y, t), u)(xij, yij, t0)*np.ones((N, N))

        return sp.lambdify((x, y, t), u)(xij, yij, t0)

    def initialize(self, N: int, mx: int, my: int) -> tuple[np.ndarray, np.ndarray]:
        r"""Initialize the solution at $U^{n}$ and $U^{n-1}$

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        mx, my : int
            Parameters for the standing wave

        Returns
        -------
        $(U^{n}, U^{n-1})$ : tuple[ndarray, ndarray]
            Initialized $U^{n}$ and $U^{n-1}$
        """
        # The docstring should probably say $U^{1}$ and $U^{0}$ instead
        xij, yij = self.create_mesh(N, sparse=True)

        U1, U0 = np.zeros((2, N+1, N+1))
        U0[:] = sp.lambdify((x, y, t), self.ue(mx, my))(xij, yij, 0)
        U1[:] = U0[:] + 0.5*(self.c*self.dt)**2*(self.D2(N) @ U0 + U0 @ self.D2(N).T)
        return U1, U0

    @property
    def dx(self) -> float:
        """Return the spatial step"""
        return 1/self.N

    @property
    def dt(self) -> float:
        """Return the time step"""
        return self.cfl*self.dx/self.c

    @property
    def L(self) -> float:
        """Return spatial length of full spatial interval"""
        return 1

    def l2_error(self, u: np.ndarray, t0: float) -> float:
        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """
        N = u.shape[0] - 1
        xij, yij = self.create_mesh(N)
        return (1/N)*np.linalg.norm(
            u - self.meshfunction(
                self.ue(self.mx, self.my), xij, yij, t0
            )
        )

    def apply_bcs(self, U: np.ndarray):
        """Apply boundary conditions to the solution mesh function

        Parameters
        ----------
        u : array
            The solution mesh function
        """
        # Setting homogeneous Dirichlet boundary conditions
        U[:, 0] = 0
        U[:, -1] = 0
        U[0, :] = 0
        U[-1, :] = 0
        return U

    def __call__(
        self,
        N: int,
        Nt: int,
        cfl: float = 0.5,
        c: float = 1.0,
        mx: int = 3,
        my: int = 3,
        store_data: int = -1,
    ):
        """Solve the wave equation

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Nt : int
            Number of time steps
        cfl : number
            The CFL number
        c : number
            The wave speed
        mx, my : int
            Parameters for the standing wave
        store_data : int
            Store the solution every store_data time step
            Note that if store_data is -1 then you should return the l2-error
            instead of data for plotting. This is used in `convergence_rates`.

        Returns
        -------
        If store_data > 0, then return a dictionary with key, value = timestep, solution
        If store_data == -1, then return the two-tuple (h, l2-error)
        """
        self.N = N
        self.Nt = Nt
        self.cfl = cfl
        self.c = c
        self.mx = mx
        self.my = my

        Un, Unm1 = self.initialize(N, mx, my)

        if store_data > 0:
            solutions = {0: Unm1.copy()}
            if store_data == 1:
                solutions[1] = Un.copy()
        elif store_data == -1:
            l2_err = []

        D2 = self.D2(N)

        # These range values makes adding solutions to dict easier
        for t in range(2, Nt+1):
            Unp1 = 2*Un - Unm1 + (c*self.dt)**2*(D2@Un + Un@(D2.T))
            Unp1 = self.apply_bcs(Unp1)
            # not t%store_data is True if t%store_data == 0 because 0 is falsy
            if store_data > 0 and not t%store_data:
                solutions[t] = Unp1.copy()
            elif store_data == -1:
                l2_err.append(self.l2_error(Unp1, 0))
            Unm1 = Un
            Un = Unp1

        if store_data > 0:
            return solutions
        elif store_data == -1:
            h = self.dt
            return h, l2_err

    def convergence_rates(
        self, m: int = 4, cfl: float = 0.1, Nt: int = 10, mx: int = 3, my: int = 3
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretizations to use
        cfl : number
            The CFL number
        Nt : int
            The number of time steps to take
        mx, my : int
            Parameters for the standing wave

        Returns
        -------
        3-tuple of arrays. The arrays represent:
            0: the orders
            1: the l2-errors
            2: the mesh sizes
        """
        E = []
        h = []
        N0 = 8
        for _ in range(m):
            dx, err = self(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
            E.append(err[-1])
            h.append(dx)
            N0 *= 2
            Nt *= 2
        r = [
            np.log(E[i - 1] / E[i]) / np.log(h[i - 1] / h[i])
            for i in range(1, m, 1)
        ]
        return np.array(r), np.array(E), np.array(h)


class Wave2D_Neumann(Wave2D):
    def D2(self, N: int) -> sparse.lil_matrix:
        raise NotImplementedError("The D2 method is not implemented yet.")

    def ue(self, mx: int, my: int) -> sp.Expr:
        raise NotImplementedError("The ue method is not implemented yet.")

    def apply_bcs(self, u: np.ndarray):
        raise NotImplementedError("The apply_bcs method is not implemented yet.")


def test_convergence_wave2d():
    sol = Wave2D()
    r, _, _ = sol.convergence_rates(m=5, mx=2, my=3)
    assert abs(r[-1] - 2) < 1e-2, r


def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann()
    r, _, _ = solN.convergence_rates(mx=3, my=3)
    assert abs(r[-1] - 2) < 0.05


def test_exact_wave2d():
    raise NotImplementedError("The test_exact_wave2d function is not implemented yet.")

if __name__ == "__main__":
    test_convergence_wave2d()