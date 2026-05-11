from ..Open import Open


class FixedPointIteration(Open):
    def __init__(self, is_verbose=False):
        super().__init__(is_verbose)
        self._gfunction = None

    @property
    def gfunction(self):
        return self._gfunction

    @gfunction.setter
    def gfunction(self, value):
        self._validate_function(value,check_arg=False)
        self._gfunction = value

    def calculate(self, x0, ea_tol=1e-6, res_tol=1e-6, max_iter=100):
        """
        Fixed Point Iteration for solving x = g(x).

        Parameters:
            x0       : Initial guess
            ea_tol   : Approximate relative error tolerance
            res_tol  : Residual tolerance |f(x)| (if self.function is provided)
            max_iter : Maximum number of iterations

        Returns:
            Approximate root
        """

        if self.gfunction is None:
            raise ValueError("gfunction is not set.")

        x = x0

        gx = self.gfunction(x)
        fx = self.function(x) if self.function is not None else gx - x

        self.add_iterations(
            {
                "iteration": 0,
                "root": x,
                "y": fx,
                "ea": "NaN",
                "|ea|": "NaN",
                "|ea|%": "NaN",
            }
        )

        for iteration in range(1, max_iter + 1):
            new_x = self.gfunction(x)

            fx = self.function(new_x) if self.function is not None else self.gfunction(new_x) - new_x

            if new_x != 0:
                ea = (new_x - x) / new_x
            else:
                ea = float("inf")

            abs_ea = abs(ea)
            abs_ea_percent = abs_ea * 100

            self.add_iterations(
                {
                    "iteration": iteration,
                    "root": new_x,
                    "y": fx,
                    "ea": ea,
                    "|ea|": abs_ea,
                    "|ea|%": abs_ea_percent,
                }
            )

            if abs_ea < ea_tol or abs(fx) < res_tol:
                return new_x

            x = new_x

        return x


if __name__ == "__main__":
    import math

    def f(x):
        return x**3 - x - 2

    def g(x):
        return (x + 2) ** (1 / 3)

    fp = FixedPointIteration()
    fp.function = f
    fp.gfunction = g

    root = fp.calculate(x0=1.5, ea_tol=1e-6)

    print(f"\nRoot found: {root:.6f}")
    print(f"f(root)   : {f(root):.2e}")
    print(fp.get_iterations())