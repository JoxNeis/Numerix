from ..Open import Open
from typing import Callable

class NewtonRaphson(Open):
    def __init__(self, is_verbose=False):
        super().__init__(is_verbose)
        self._derivative = None

    @property
    def derivative(self):
        return self._derivative

    @derivative.setter
    def derivative(self, value: Callable):
        self._validate_function(value, check_arg=False)
        self._derivative = value

    def calculate(self, x0, ea_tol=1e-6, res_tol=1e-6, max_iter=100):
        """
        Newton-Raphson method for solving f(x) = 0.

        Formula:
            x_{n+1} = x_n - f(x_n) / f'(x_n)

        Parameters:
            x0       : Initial guess
            ea_tol   : Approximate relative error tolerance
            res_tol  : Residual tolerance |f(x)|
            max_iter : Maximum number of iterations

        Returns:
            Approximate root
        """

        if self.function is None:
            raise ValueError("function is not set.")

        if self.derivative is None:
            raise ValueError("derivative is not set.")

        x = x0
        fx = self.function(x)

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
            dfx = self.derivative(x)

            if dfx == 0:
                raise ZeroDivisionError(
                    f"Derivative is zero at x = {x}. Newton-Raphson cannot continue."
                )

            new_x = x - fx / dfx

            fx = self.function(new_x)

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

    def f(x):
        return x**3 - x - 2

    def df(x):
        return 3 * x**2 - 1

    nr = NewtonRaphson()
    nr.function = f
    nr.derivative = df

    root = nr.calculate(x0=1.5, ea_tol=1e-6)

    print(f"\nRoot found: {root:.6f}")
    print(f"f(root)   : {f(root):.2e}")
    print(nr.get_iterations())
