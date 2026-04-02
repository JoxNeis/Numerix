import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import inspect
import math

from typing import Dict, Any, Callable


# region VISUALIZER
class Visualizer:
    def __init__(self):
        self._fig, self._axes = plt.subplots()

    # region LINESPACE
    def _create_linespace(
        self,
        function: Callable,
        begin: float,
        end: float,
        offset: float,
        data_points: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        begin -= offset
        end += offset

        points = (
            int(data_points) if data_points > 0 else max(int((end - begin) * 10), 400)
        )

        x = np.linspace(begin, end, points)
        y = function(x)

        return x, y

    def _create_linespaces(
        self,
        functions: list[Callable],
        begin: float,
        end: float,
        offset: float = 5,
        data_points: int = 0,
    ) -> list[tuple[np.ndarray, np.ndarray]]:

        return [
            self._create_linespace(
                func,
                begin,
                end,
                offset,
                data_points,
            )
            for func in functions
        ]

    # endregion
    # region PLOTTING
    # region CARTESIAN
    def __create_cartesian_plane(
        self,
        axis_color="black",
        axis_width=1,
        grid_style="--",
        grid_width=0.5,
        grid_transparency=0.4,
    ) -> None:
        self._axes.axhline(0, color=axis_color, linewidth=axis_width, label="X")
        self._axes.axvline(0, color=axis_color, linewidth=axis_width, label="Y")

        self._axes.grid(
            True,
            linestyle=grid_style,
            linewidth=grid_width,
            alpha=grid_transparency,
        )

    def custom_cartesian_plane(
        self,
        axis_color="black",
        axis_width=1,
        grid_style="--",
        grid_width=0.5,
        grid_transparency=0.4,
    ) -> None:
        self.__create_cartesian_plane(
            axis_color, axis_width, grid_style, grid_width, grid_transparency
        )

    # endregion
    # region GRAPH
    def __get_symbol_name(
        self,
        function: Callable,
        index: int,
    ) -> str:

        name = getattr(function, "__name__", None)

        if not name or name == "<lambda>":
            name = f"f{index}"

        return name

    def __plot_functions(
        self,
        linespaces: list[tuple[np.ndarray, np.ndarray]],
        functions: list[Callable],
    ) -> None:

        for i, ((x, y), func) in enumerate(
            zip(linespaces, functions),
            start=1,
        ):

            name = self.__get_symbol_name(func, i)

            self._axes.plot(
                x,
                y,
                label=f"{name}(x)",
                linewidth=1.5,
            )

    def create_graph(
        self,
        functions: list[Callable],
        begin: float,
        end: float,
        offset: float = 5,
        data_points: int = 0,
        cartesian_plane: bool = True,
    ):

        linespaces = self._create_linespaces(
            functions,
            begin,
            end,
            offset,
            data_points,
        )

        self.__plot_functions(
            linespaces,
            functions,
        )

        if cartesian_plane:
            self.__create_cartesian_plane()
        self._axes.legend()

        return self._axes

    # endregion
    # region GRAPH PROPERTIES
    def create_anotation(self):
        pass

    def create_dot(self):
        pass

    def create_vertical_span(self):
        pass

    def create_vertical_line(self):
        pass

    def create_horizontal_line(self):
        pass

    def create_horizontal_span(self):
        pass

    # endregion


    # region ANIMATION

    # endregion
    #endregion
    # region SHOW
    def show(self) -> None:
        self._fig.tight_layout()
        plt.show()

    def clear(self) -> None:
        self._axes.cla()

    # endregion
# endregion

# region Numerix
class Numerix:
    """
    Base class for numerical method implementations.
    """

    def __init__(self, is_verbose: bool = False):

        self._is_verbose = is_verbose
        self.__visualizer = Visualizer()

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
            self.iterations = pd.DataFrame([iteration])
        else:
            self.iterations.loc[len(self.iterations)] = iteration
        self._after_each_iteration(iteration)

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

    # endregion

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


    def plot_result(self, axis_color="black"):
        if not (len(self.iterations) > 0):
            raise RuntimeError("No data to plot.")
        self._create_linespaces_for_all()
        self._create_plot_from_linespace()
        self._create_plot_result_point()

        plt.axhline(0, color=axis_color, label="X")
        plt.axvline(0, color=axis_color, label="Y")
        plt.grid(True)
        plt.show()

    def _create_animation_plot_from_linespace(self, ax):
        self._line_artists = []

        for i, ((x, y), func) in enumerate(zip(self.__linespaces, self.functions)):
            name = getattr(func, "__name__", f"f{i+1}")

            if name == "<lambda>":
                name = f"f{i+1}"
            (line,) = ax.plot(x, y, label=f"{name}(x)")
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
            blit=False,
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

    # region Properties
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

    # endregion

    # region Visualization
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

        if hasattr(self, "_current_artists"):
            for artist in self._current_artists:
                artist.remove()

        self._current_artists = []

        span = ax.axvspan(lower, upper, alpha=0.2)

        lower_line = ax.axvline(
            lower,
            linestyle="-",
        )

        upper_line = ax.axvline(
            upper,
            linestyle="-",
        )

        mid_line = ax.axvline(midpoint, linestyle="--")

        mid_dot = ax.plot(midpoint, f_mid, marker="o")[0]

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
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        self._current_artists.extend(
            [span, lower_line, upper_line, mid_line, mid_dot, iter_text]
        )

        return self._current_artists

    # endregion

    def calculate(self):
        pass


# endregion
