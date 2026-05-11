from ..Open import Open

class Secant(Open):
    def __init__(self, is_verbose=False):
        super().__init__(is_verbose)

    def calculate(self, x0, x1, ea_tol=1e-6, res_tol=1e-6, max_iter=100):
        """
        Secant method for solving f(x) = 0.

        Parameters:
            x0, x1   : Initial guesses
            ea_tol   : Approximate relative error tolerance
            res_tol  : Residual tolerance |f(x)|
            max_iter : Maximum number of iterations

        Returns:
            Approximate root
        """

        f0 = self.function(x0)
        f1 = self.function(x1)

        self.add_iterations(
            {
                "iteration": 0,
                "root": x0,
                "y": f0,
                "ea": "NaN",
                "|ea|": "NaN",
                "|ea|%": "NaN",
            }
        )

        self.add_iterations(
            {
                "iteration": 1,
                "root": x1,
                "y": f1,
                "ea": "NaN",
                "|ea|": "NaN",
                "|ea|%": "NaN",
            }
        )

        for iteration in range(2, max_iter + 1):
            if f1 == f0:
                raise ZeroDivisionError(
                    "Secant method failed because f(x1) == f(x0)."
                )

            x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
            f2 = self.function(x2)

            if x2 != 0:
                ea = (x2 - x1) / x2
            else:
                ea = float("inf")

            abs_ea = abs(ea)
            abs_ea_percent = abs_ea * 100

            self.add_iterations(
                {
                    "iteration": iteration,
                    "root": x2,
                    "y": f2,
                    "ea": ea,
                    "|ea|": abs_ea,
                    "|ea|%": abs_ea_percent,
                }
            )

            if abs_ea < ea_tol or abs(f2) < res_tol:
                return x2

            x0, x1 = x1, x2
            f0, f1 = f1, f2

        Warning(f"Method did not converge within {max_iter} iterations")
        return x2
    
if __name__ == "__main__":
    import math

    def f(x):
        return x**3 - x - 2

    s = Secant()
    s.add_function(f)

    root = s.calculate(x0=1, x1=2, ea_tol=1e-6)

    print(f"\nRoot found: {root:.6f}")
    print(f"f(root)   : {f(root):.2e}")
    print(s.get_iterations())