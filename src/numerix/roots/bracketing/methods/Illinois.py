from ..Bracketing import Bracketing


class Illinois(Bracketing):
    def __init__(self, is_verbose=False):
        super().__init__(is_verbose)

    def calculate(self, low, upper, ea_tol=1e-6, res_tol=1e-6, max_iter=100):
        """
        Find root using the Illinois Method.

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
            Residual tolerance — stops when |f(xr)| < res_tol.
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
        xr = None
        changed = None

        f_low = self.function(low)
        f_upper = self.function(upper)

        for iteration in range(max_iter):
            old_xr = xr
            xr = upper - (f_upper * (low - upper) / (f_low - f_upper))
            f_xr = self.function(xr)

            if f_low * f_xr < 0:
                upper = xr
                if changed == "upper":
                    f_low = f_low / 2
                f_upper = f_xr
                changed = "upper"
            else:
                low = xr
                if changed == "low":
                    f_upper = f_upper / 2

                f_low = f_xr
                changed = "low"

            if old_xr is None or xr == 0:
                ea = float("inf")
            else:
                ea = abs((xr - old_xr) / xr)

            self.add_iterations(
                {
                    "iteration": iteration + 1,
                    "low": low,
                    "upper": upper,
                    "xr": xr,
                    "f_low": f_low,
                    "f_xr": f_xr,
                    "f_upper": f_upper,
                    "ea": ea,
                    "|ea|": abs(ea),
                    "|ea|%": abs(ea) * 100,
                }
            )
            
            if ea <= ea_tol or abs(f_xr) < res_tol:
                if self._is_verbose:
                    print(f"Illinois converged with: {xr}")
                return xr

        print(
                f"Illinois method did not converge after {max_iter} iterations."
            )

if __name__ == "__main__":
    import math

    def f(x):
        return x**3 - x - 2

    b = Illinois()
    b.add_function(f)
    root = b.calculate(low=1, upper=2, ea_tol=1e-6)

    print(f"\nRoot found: {root:.6f}")
    print(f"f(root)   : {f(root):.2e}")
    print(b.get_iterations())
