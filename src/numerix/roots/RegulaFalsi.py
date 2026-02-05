from typing import Callable
from numerix.Numerix import Numerix


class RegulaFalsi(Numerix):
    """
    Regula Falsi (False Position) method for finding roots of a
    continuous single-variable function.
    """

    def __init__(
        self,
        function: Callable,
        initial_lower_bound: float,
        initial_upper_bound: float,
        *,
        is_verbose: bool = False,
        is_logging: bool = False,
    ):
        super().__init__(is_verbose=is_verbose, is_logging=is_logging)
        self.set_function(function)
        self.set_bounds(initial_lower_bound, initial_upper_bound)

    def set_function(self, function: Callable):
        if not callable(function):
            raise TypeError("Function must be callable.")

        self.function = function
        self.args_count = function.__code__.co_argcount

        if self.args_count != 1:
            raise ValueError(
                "Regula Falsi supports only single-variable functions f(x)."
            )

    def set_bounds(self, lower: float, upper: float):
        if lower >= upper:
            raise ValueError("Lower bound must be less than upper bound.")

        self.lower = float(lower)
        self.upper = float(upper)

    def start(self, tol: float = 1e-6, max_iter: int = 1000) -> float:
        f = self.function
        a, b = self.lower, self.upper
        fa, fb = f(a), f(b)

        if fa * fb >= 0:
            raise ValueError(
                "Function must have opposite signs at the bounds."
            )

        for i in range(1, max_iter + 1):
            # Regula Falsi formula
            c = b - fb * (b - a) / (fb - fa)
            fc = f(c)

            if self.is_logging:
                self.add_logs(
                    {
                        "iter": i,
                        "lower": a,
                        "upper": b,
                        "c": c,
                        "f(c)": fc,
                    }
                )

            if abs(fc) < tol:
                return c

            if fa * fc < 0:
                b, fb = c, fc
            else:
                a, fa = c, fc

        raise RuntimeError("Regula Falsi method did not converge.")
