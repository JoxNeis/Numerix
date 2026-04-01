import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import inspect
import random

from typing import Dict, Any, Callable


# region Numerix
class Numerix:
    """
    Base class for numerical method implementations.
    """

    def __init__(self, is_verbose: bool = False):

        self._is_verbose = is_verbose

        self._columns: list[str] = []
        self.__linespaces: list[tuple[np.ndarray, np.ndarray]] = []

        self._min_x: float = None
        self._max_x: float = None

        self.iterations = pd.DataFrame()

        self.functions: list[Callable] = []
        self.argument_count: int | None = None

    # region Properties
    def _challenge_min_max_x(self, iteration):
        pass

    def _process_on_each_iterations(self, iteration: Dict[str, Any]):
        # to be overriden
        pass

    def add_iterations(self, iteration: Dict[str, Any]):
        if self.iterations.empty:
            self._columns = list(iteration.keys())
            self.iterations = pd.DataFrame([iteration])
        else:
            self.iterations.loc[len(self.iterations)] = iteration

        self._challenge_min_max_x(iteration)

        if self._is_verbose:
            print(iteration)

    def add_function(self, function: Callable):
        """
        Store mathematical function and enforce
        consistent argument count.
        """

        if not callable(function):
            raise TypeError("Function must be callable.")

        signature = inspect.signature(function)
        arg_count = len(signature.parameters)

        if not hasattr(self, "argument_count"):
            self.argument_count = arg_count

        if arg_count != self.argument_count:
            raise ValueError(
                "Function argument count mismatch. "
                f"Expected {self.argument_count}, "
                f"got {arg_count}."
            )

        self.functions.append(function)

        if self._is_verbose:
            print(f"Function added with " f"{arg_count} argument(s).")

    # region Calculation
    def get_tolerance_from_significant_digit(self, digits: int) -> float:

        if digits <= 0:
            raise ValueError("Significant digits must be positive.")

        tolerance = 0.5 * 10 ** (-digits)
        return tolerance

    def calculate(self):
        # to be overridden
        pass

    # region Visualization
    def display_iterations(self):
        if self.iterations.empty:
            print("No iterations recorded.")
            return

        print(self.iterations)

    def __create_linespace(self, function: Callable, display_scale=2):
        if self._min_x is None or self._max_x is None:
            return
        begin = self._min_x * display_scale
        end = self._max_x * display_scale
        x = np.linspace(begin, end)
        y = function(x)
        self.__linespaces.append((x, y))

    def __create_linespaces_for_all(self):
        self.__linespaces.clear()

        for function in self.functions:
            self.__create_linespace(function)

    def __create_plot_from_linespace(self):
        for i, ((x, y), func) in enumerate(zip(self.__linespaces, self.functions)):
            name = getattr(func, "__name__", f"f{i+1}")
            if name == "<lambda>":
                name = f"f{i+1}"
            plt.plot(x, y, label=f"{name}(x)")

    def draw_iteration_points(self):
        pass

    def plot_iterations(self, **plot_kwargs):
        if not self.functions:
            raise RuntimeError("No functions added.")

        self.__create_linespaces_for_all()
        self.__create_plot_from_linespace()
        self.draw_iteration_points()

        plt.axhline(0)
        plt.axvline(0)

        plt.grid(True)

        plt.show()


# region Bisection
class Bisection(Numerix):
    def __init__(self, _is_verbose=False):
        super().__init__(_is_verbose)
        self.boundaries: list[Dict[str, Any]] = [{}]
        self.function: Callable | None = None
        self.argument_count = 1

    def _challenge_min_max_x(self, iteration):
        if self._min_x is None:
            self._min_x = float(iteration["lower"])
        if self._max_x is None:
            self._max_x = float(iteration["upper"])
        self._min_x = min(self._min_x, float(iteration["lower"]))
        self._max_x = max(self._max_x, float(iteration["lower"]))

    def add_function(self, function: Callable):
        self.functions.clear()
        super().add_function(function)
        self.function = self.functions[0]

    def set_initial_boundary(self, lower: float, upper: float):

        if self.function is None:
            raise RuntimeError("No function defined.")

        y_low = self.function(lower)
        y_upper = self.function(upper)

        if y_low * y_upper > 0:
            raise ValueError("f(lower) and f(upper) " "must have opposite signs.")

        self.boundaries[0] = {"lower": lower, "upper": upper}

    def draw_iteration_points(self):
        for iteration in self.iterations.itertuples():
            color = (random.random(), random.random(), random.random())
            plt.axvspan(
                float(iteration.lower),
                float(iteration.upper),
                alpha=0.1,
                color=color,
                label=f"Iteration: {iteration.iteration}",
            )
            plt.plot(
                iteration.midpoint,
                iteration.f_mid,
                "o",
                color=color,
                label=f"Iteration: {iteration.iteration}",
            )
            plt.annotate(
                f"{iteration.iteration}",
                xy=(iteration.midpoint, iteration.f_mid),
                xytext=(0, 0),
                textcoords="points",
                fontsize=8,
                color='white',
            )

    def calculate(self, tolerance: float = 1e-6, max_iterations: int = 100):
        if not self.boundaries:
            raise RuntimeError("Initial boundary not set.")

        lower = self.boundaries[0]["lower"]
        upper = self.boundaries[0]["upper"]

        previous_midpoint = None

        for iteration in range(max_iterations):

            midpoint = (lower + upper) / 2

            f_low = self.function(lower)
            f_mid = self.function(midpoint)

            if previous_midpoint is None:
                ea = None
                er = None
            else:
                ea = abs(midpoint - previous_midpoint)
                if midpoint != 0:
                    er = ea / abs(midpoint)
                else:
                    er = None

            self.add_iterations(
                {
                    "iteration": iteration,
                    "lower": lower,
                    "upper": upper,
                    "midpoint": midpoint,
                    "f_mid": f_mid,
                    "ea": ea,
                    "er": er,
                    "Ea": "Unknown",
                    "Er": "Unknown",
                }
            )

            if ea is not None and ea < tolerance:
                if self._is_verbose:
                    print("Iteration stopped, reach tolerance")
                return midpoint

            if f_low * f_mid < 0:
                upper = midpoint
            else:
                lower = midpoint
            previous_midpoint = midpoint

        return midpoint


def test():
    def f(x):
        return x**2 - 4

    solver = Bisection()
    solver.add_function(f)
    solver.set_initial_boundary(lower=0, upper=3)
    root = solver.calculate()
    solver.display_iterations()
    solver.plot_iterations()


if __name__ == "__main__":
    test()
