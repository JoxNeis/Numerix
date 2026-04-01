import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import inspect
import math

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

    def _after_each_iteration(self, iteration: Dict[str, Any]):
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

    # endregion

    # region Calculation
    @staticmethod
    def get_tolerance_from_significant_digit(digits: int) -> float:
        if digits <= 0:
            raise ValueError("Significant digits must be positive.")

        tolerance = 0.5 * 10 ** (-digits)
        return tolerance

    def calculate(self):
        # to be overridden
        pass

    # endregion

    # region Results
    def display_iterations(self):
        if self.iterations.empty:
            print("No iterations recorded.")
            return
        print(self.iterations)

    # region Visualization
    def __create_linespace(self, function: Callable, offset=5, display_scale=1):
        if self._min_x is None or self._max_x is None:
            return
        begin = math.floor(self._min_x) * display_scale - offset
        end = math.ceil(self._max_x * display_scale) + offset
        x = np.linspace(begin, end)
        y = function(x)
        self.__linespaces.append((x, y))

    def _create_linespaces_for_all(self):
        self.__linespaces.clear()

        for function in self.functions:
            self.__create_linespace(function)

    def _create_plot_from_linespace(self):
        for i, ((x, y), func) in enumerate(zip(self.__linespaces, self.functions)):
            name = getattr(func, "__name__", f"f{i+1}")
            if name == "<lambda>":
                name = f"f{i+1}"
            plt.plot(x, y, label=f"{name}(x)")

    def _create_plot_result_point(self):
        pass

    def plot_result(self,axis_color="black"): 
        if not (len(self.iterations) > 0): 
            raise RuntimeError("No data to plot.") 
        self._create_linespaces_for_all() 
        self._create_plot_from_linespace() 
        self._create_plot_result_point() 
        
        plt.axhline(0,color=axis_color,label="X") 
        plt.axvline(0,color=axis_color,label="Y") 
        plt.grid(True) 
        plt.show()
        
    def _create_animation_plot_from_linespace(self, ax):
        self._line_artists = []

        for i, ((x, y), func) in enumerate(
            zip(self.__linespaces, self.functions)
        ):
            name = getattr(func, "__name__", f"f{i+1}")

            if name == "<lambda>":
                name = f"f{i+1}"
            line, = ax.plot(x, y, label=f"{name}(x)")
            self._line_artists.append(line)
            
    def animate_iterations(self, axis_color="black", interval=400):
        if not (len(self.iterations) > 0):
            raise RuntimeError("No data to animate.")

        fig, ax = plt.subplots()
        self._ax = ax

        self._create_linespaces_for_all()
        self._create_animation_plot_from_linespace(ax)

        ax.axhline(0, color=axis_color)
        ax.axvline(0, color=axis_color)

        ax.grid(True)
        ax.legend()

        self._current_artists = []

        ani = animation.FuncAnimation(
            fig,
            self._update,
            frames=len(self.iterations),
            interval=interval,
            blit=False
        )

        plt.show()  

    # endregion
    # endregion


# endregion


# region Bracketing
class Bracketing(Numerix):
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
        self._max_x = max(self._max_x, float(iteration["upper"]))

    def _after_each_iteration(self, iteration):
        self._challenge_min_max_x(iteration)

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

    def __draw_boundary_line(self, lower, upper, color, alpha, linewidth):
        plt.axvline(lower, color=color, alpha=alpha, linewidth=linewidth)
        plt.axvline(upper, color=color, alpha=alpha, linewidth=linewidth)

    def _create_plot_result_point(
        self,
        iteration_color="red",
        result_color="lime",
        iteration_alpha=1,
        result_alpha=1,
        iteration_linewidth=1,
        result_linewidth=1,
        text_offset=(0, 10),
        dot_size=5,
        text_size=6,
    ):
        if len(self.iterations) == 0:
            return

        total = len(self.iterations)

        for i, iteration in enumerate(self.iterations.itertuples()):
            if i < (total - 1):
                self.__draw_boundary_line(
                    iteration.lower,
                    iteration.upper,
                    iteration_color,
                    iteration_alpha,
                    iteration_linewidth,
                )

        last = self.iterations.iloc[-1]

        plt.axvline(
            last.midpoint,
            color=result_color,
            alpha=result_alpha,
            linewidth=result_linewidth,
        )
        plt.plot(
            last.midpoint,
            last.f_mid,
            "o",
            color=result_color,
            markersize=dot_size,
            alpha=result_alpha,
        )
        plt.annotate(
            f"x ≈ {last.midpoint:.2f}",
            xy=(last.midpoint, last.f_mid),
            xytext=text_offset,
            textcoords="offset points",
            fontsize=text_size,
            color="black",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=1),
        )

    def _update(self, frame):
        row = self.iterations.iloc[frame]

        lower = row["lower"]
        upper = row["upper"]
        midpoint = row["midpoint"]
        f_mid = row["f_mid"]

        ax = self._ax

        # Remove previous artists
        if hasattr(self, "_current_artists"):
            for artist in self._current_artists:
                artist.remove()

        self._current_artists = []

        # Draw span
        span = ax.axvspan(
            lower,
            upper,
            alpha=0.2
        )

        # Lower bound line
        lower_line = ax.axvline(
            lower,
            linestyle="-",
        )

        # Upper bound line
        upper_line = ax.axvline(
            upper,
            linestyle="-",
        )

        # Midpoint line
        mid_line = ax.axvline(
            midpoint,
            linestyle="--"
        )

        # Midpoint dot
        mid_dot = ax.plot(
            midpoint,
            f_mid,
            marker="o"
        )[0]

        # Iteration info box
        iter_text = ax.text(
            0.02,
            0.95,
            (
                f"Iteration: {frame}/{len(self.iterations)}\n"
                f"Lower: {lower:.4f}\n"
                f"Upper: {upper:.4f}\n"
                f"Mid: {midpoint:.4f}"
            ),
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                alpha=0.8
            )
        )

        # Store artists
        self._current_artists.extend([
            span,
            lower_line,
            upper_line,
            mid_line,
            mid_dot,
            iter_text
        ])

        return self._current_artists
         
    def calculate(self):
        pass


# endregion
# region Bisection
class Bisection(Bracketing):
    def __init__(self,_is_verbose=False):
        super().__init__(_is_verbose)

    def calculate(self, tolerance: float = 1e-4, max_iterations: int = 100):
        if not self.boundaries:
            raise RuntimeError("Initial boundary not set.")

        lower = self.boundaries[0]["lower"]
        upper = self.boundaries[0]["upper"]

        f_low = self.function(lower)
        previous_midpoint = None

        for iteration in range(max_iterations):
            midpoint = (lower + upper) / 2
            f_mid = self.function(midpoint)

            ea = (
                abs(midpoint - previous_midpoint)
                if previous_midpoint is not None
                else None
            )
            er = (ea / abs(midpoint)) if (ea is not None and midpoint != 0) else None

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

            if f_mid == 0 or (ea is not None and ea < tolerance):
                if self._is_verbose:
                    print("Iteration stopped, reached tolerance")
                return midpoint

            if f_low * f_mid < 0:
                upper = midpoint
            else:
                lower = midpoint
                f_low = f_mid

            previous_midpoint = midpoint

        return midpoint


def test():
    def f(x):
        return x**2 - 4

    solver = Bisection()
    solver.add_function(f)
    solver.set_initial_boundary(lower=0, upper=6)
    root = solver.calculate()
    print(f"Result: {root:.4f}")
    solver.display_iterations()
    solver.animate_iterations()


if __name__ == "__main__":
    test()
