from sympy import Mul, symbols

x = symbols("x")

def Lagrangebasis(xj, x=x):
    """Construct Lagrange basis for points in xj

    Parameters
    ----------
    xj : array
    Interpolation points (nodes)
    x : Sympy Symbol

    Returns
    -------
    Lagrange basis as a list of Sympy functions
    """
    n = len(xj)
    ell = []
    numert = Mul(*[x - xj[i] for i in range(n)])
    for i in range(n):
        numer = numert/(x - xj[i])
        denom = Mul(*[(xj[i] - xj[j]) for j in range(n) if i != j])
        ell.append(numer/denom)
    return ell