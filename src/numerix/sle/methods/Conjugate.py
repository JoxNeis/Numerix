from ..Sle import Sle
import numpy as np


class Conjugate(Sle):
    def __init__(self, is_verbose=False):
        super().__init__(is_verbose)
        self._A = None
        self._b = None

    # region SYSTEM
    @property
    def A(self):
        return self._A

    @A.setter
    def A(self, matrix):
        matrix = np.array(matrix, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("A must be a 2D matrix.")
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"A must be square. Got shape {matrix.shape}."
            )
        if not np.allclose(matrix, matrix.T):
            raise ValueError("A must be symmetric.")
        if not np.all(np.linalg.eigvals(matrix) > 0):
            raise ValueError("A must be positive definite.")
        self._A = matrix

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, vector):
        vector = np.array(vector, dtype=float)
        if vector.ndim != 1:
            raise ValueError("b must be a 1D vector.")
        if self._A is not None and len(vector) != self._A.shape[0]:
            raise ValueError(
                f"b length {len(vector)} does not match "
                f"A rows {self._A.shape[0]}."
            )
        self._b = vector

    def set_system(self, A, b):
        self.A = A
        self.b = b

    # endregion

    # region CALCULATE
    def calculate(self, ea_tol=1e-6, max_iter=100):
        if self._A is None or self._b is None:
            raise ValueError(
                "System not set. Call set_system(A, b) first."
            )

        A, b = self._A, self._b
        n = len(b)

        x     = np.zeros(n)
        r     = b - A @ x
        p     = r.copy()
        r_dot = r @ r

        self.iterations = []

        for k in range(max_iter):
            Ap      = A @ p
            alpha   = r_dot / (p @ Ap)
            x_new   = x + alpha * p
            r_new   = r - alpha * Ap
            r_dot_new = r_new @ r_new

            beta    = r_dot_new / r_dot
            p       = r_new + beta * p

            with np.errstate(divide='ignore', invalid='ignore'):
                ea_vec = np.where(
                    x_new != 0,
                    np.abs((x_new - x) / x_new),
                    np.abs(x_new - x)
                )
            ea = float(np.max(ea_vec))

            self.iterations.append({
                "iteration" : k + 1,
                "args"      : tuple(x_new),
                "|ea|"      : ea,
                "|ea|%"     : ea * 100,
                "residual"  : float(np.linalg.norm(r_new)),
            })

            x     = x_new
            r     = r_new
            r_dot = r_dot_new

            if ea < ea_tol:
                return tuple(x)

        Warning(f"Method did not converge within {max_iter} iterations")
        return tuple(x)
    # endregion

if __name__ == "__main__":
    # 10x - y + 2z = 6
    # -x + 11y - z = 25
    #  2x - y + 10z = -11
    A = [
        [10, -1,  2],
        [-1, 11, -1],
        [ 2, -1, 10],
    ]
    b = [6, 25, -11]

    conjugate = Conjugate()
    conjugate.set_system(A, b)

    solution = conjugate.calculate(ea_tol=1e-6)

    print("\nSolution found:")
    x, y, z = solution
    print(f"x = {x:.6f}")
    print(f"y = {y:.6f}")
    print(f"z = {z:.6f}")

    print("\nVerification:")
    print(f"10x -  y + 2z = {10*x -   y + 2*z:.6f}  (should be   6)")
    print(f" -x + 11y -  z = {  -x + 11*y -   z:.6f}  (should be  25)")
    print(f" 2x -  y + 10z = { 2*x -   y + 10*z:.6f}  (should be -11)")

    print("\nIterations:")
    from tabulate import tabulate
    print(tabulate(conjugate.iterations, headers="keys", tablefmt="psql"))