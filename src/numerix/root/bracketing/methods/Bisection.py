from ..Bracketing import Bracketing

class Bisection(Bracketing):
    def __init__(self, is_verbose=False):
        super().__init__(is_verbose)

    def calculate(self, low, upper, ea_tol=1e-6, res_tol=1e-6, max_iter=100):
        """
        Find root using the Bisection Method.

        Parameters
        ----------
        low : float
            Lower bound of interval
        upper : float
            Upper bound of interval
        ea_tol : float
            Approximate relative error tolerance (stopping condition).
            Iteration stops when |ea| <= ea_tol.
        res_tol : float, optional
            Residual tolerance — stops when |f(mid)| < res_tol.
            Default is 1e-6.
        max_iter : int, optional
            Maximum number of iterations. Default is 100.

        Returns
        -------
        float
            Approximated root

        Raises
        ------
        ValueError
            If initial interval does not bracket a root
        RuntimeError
            If method fails to converge
        """
        self._check_brackets(low, upper)
        mid = None

        for iteration in range(max_iter):
            old_mid = mid
            mid = (low + upper) / 2

            f_low = self.function(low)
            f_mid = self.function(mid)
            f_upper = self.function(upper)

            if f_low * f_mid < 0:
                upper = mid
            else:
                low = mid

            if old_mid is None or mid == 0:
                ea = float("inf")
            else:
                ea = (mid - old_mid) / mid

            self.add_iterations(
                {
                    "iteration": iteration + 1,
                    "low": low,
                    "upper": upper,
                    "mid": mid,
                    "f_low": f_low,
                    "f_mid": f_mid,
                    "f_upper": f_upper,
                    "new_low": low,
                    "new_upper": upper,
                    "ea": ea,
                    "|ea|": abs(ea),
                    "|ea|%": abs(ea) * 100, 
                }
            )

            if abs(ea) <= ea_tol or abs(f_mid) < res_tol:
                if(self._is_verbose):
                    print(f"Bisection converged with: {mid}")
                return mid

        print(
            f"Bisection method did not converge after {max_iter} iterations."
        )
        
if __name__ == "__main__":
    import math

    def f(x):
        return x**3 - x - 2

    b = Bisection()
    b.add_function(f)
    root = b.calculate(low=1, upper=2, ea_tol=1e-6)

    print(f"\nRoot found: {root:.6f}")
    print(f"f(root)   : {f(root):.2e}")
    print(b.get_iterations())