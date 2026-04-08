import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Any, Callable


# region VISUALIZER
class Visualizer:
    """
    Base class for function visualizations with optional iteration plotting and animation.

    Subclasses should override:
        - _find_range()       : sets self.begin and self.end from self._iterations / self._functions
        - _plot_iteration()   : draws artists for a single iteration row onto self._axes
    """

    def __init__(
        self,
        iterations: pd.DataFrame,
        functions: list[Callable],
    ) -> None:
        self._fig, self._axes = plt.subplots()

        self._iterations: pd.DataFrame = iterations
        self._functions: list[Callable] = functions

        # Holds artists added per-iteration so they can be cleared between frames
        self._iteration_artists: list = []
        self._ani: animation.FuncAnimation | None = None

        self.begin: float = 0.0
        self.end: float = 0.0

        self._find_range()

    # region LINESPACE
    def _find_range(self) -> None:
        """
        Override in subclasses to set self.begin and self.end.

        Example implementation:
            x_col = self._iterations["x"]
            self.begin = float(x_col.min())
            self.end   = float(x_col.max())
        """
        pass

    def _create_linespace(
        self,
        function: Callable,
        offset: float,
        data_points: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (x, y) arrays for *function* over [begin-offset, end+offset]."""
        begin = self.begin - offset
        end = self.end + offset

        points = (
            int(data_points) if data_points > 0 else max(int((end - begin) * 10), 400)
        )

        x = np.linspace(begin, end, points)
        y = function(x)
        return x, y

    def _create_linespaces(
        self,
        offset: float = 5,
        data_points: int = 0,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return a list of (x, y) arrays for every function in self._functions."""
        return [
            self._create_linespace(func, offset, data_points)
            for func in self._functions
        ]

    # endregion

    # region PLOTTING

    # region CARTESIAN PLANE
    def __create_cartesian_plane(
        self,
        axis_color: str = "black",
        axis_width: float = 1,
        grid_style: str = "--",
        grid_width: float = 0.5,
        grid_transparency: float = 0.4,
    ) -> None:
        self._axes.axhline(0, color=axis_color, linewidth=axis_width)
        self._axes.axvline(0, color=axis_color, linewidth=axis_width)
        self._axes.grid(
            True,
            linestyle=grid_style,
            linewidth=grid_width,
            alpha=grid_transparency,
        )

    def custom_cartesian_plane(
        self,
        axis_color: str = "black",
        axis_width: float = 1,
        grid_style: str = "--",
        grid_width: float = 0.5,
        grid_transparency: float = 0.4,
    ) -> None:
        """Public wrapper so subclasses / callers can draw the Cartesian plane."""
        self.__create_cartesian_plane(
            axis_color, axis_width, grid_style, grid_width, grid_transparency
        )

    # endregion

    # region GRAPH DECORATORS
    def add_title(self, title: str) -> None:
        self._axes.set_title(title)

    def create_annotation(
        self,
        text: str,
        xy: tuple[float, float],
        offset: tuple[float, float] = (5, 5),
        text_size: int = 10,
    ) -> None:
        """Add a boxed text annotation pointing at *xy*."""
        self._axes.annotate(
            text,
            xy=xy,
            xytext=offset,
            textcoords="offset points",
            fontsize=text_size,
            color="black",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=1),
        )

    # Keep the original (typo) name as an alias for backwards compatibility
    def create_anotation(self, text, xy, offset=(5, 5), text_size=10):
        self.create_annotation(text, xy, offset, text_size)

    def create_dot(
        self,
        x: float,
        y: float,
        color: str = "lime",
        dot_size: float = 5,
        alpha: float = 1.0,
    ) -> None:
        self._axes.plot(x, y, "o", color=color, markersize=dot_size, alpha=alpha)

    def create_vertical_line(
        self, x: float, color: str = "lime", alpha: float = 0.8, linewidth: float = 0.8
    ) -> None:
        self._axes.axvline(x, color=color, alpha=alpha, linewidth=linewidth)

    def create_horizontal_line(
        self, y: float, color: str = "lime", alpha: float = 0.8, linewidth: float = 0.8
    ) -> None:
        self._axes.axhline(y, color=color, alpha=alpha, linewidth=linewidth)

    def create_vertical_span(
        self,
        left: float,
        right: float,
        alpha: float = 0.2,
        linewidth: float = 1,
        color: str = "lime",
    ) -> None:
        self._axes.axvspan(left, right, alpha=alpha, linewidth=linewidth, color=color)

    def create_horizontal_span(
        self,
        bottom: float,
        top: float,
        alpha: float = 0.2,
        linewidth: float = 1,
        color: str = "lime",
    ) -> None:
        self._axes.axhspan(bottom, top, alpha=alpha, linewidth=linewidth, color=color)

    # endregion

    # region FUNCTION GRAPH
    @staticmethod
    def __get_function_name(function: Callable, index: int) -> str:
        """Return a human-readable label for *function*."""
        name = getattr(function, "__name__", None)
        if not name or name in ("<lambda>", ""):
            name = f"f{index}"
        return name

    def __plot_functions(
        self,
        linespaces: list[tuple[np.ndarray, np.ndarray]],
    ) -> None:
        for i, ((x, y), func) in enumerate(
            zip(linespaces, self._functions), start=1
        ):
            name = self.__get_function_name(func, i)
            self._axes.plot(x, y, label=f"{name}(x)", linewidth=1.5)

    def create_graph(
        self,
        title: str,
        offset: float = 5,
        data_points: int = 0,
        cartesian_plane: bool = True,
    ) -> Axes:
        """
        Plot all functions and set up the base graph.

        Parameters
        ----------
        title           : Chart title.
        offset          : Extra range added on both sides of [begin, end].
        data_points     : Number of sample points (0 = auto).
        cartesian_plane : Whether to draw x/y axes and a grid.

        Returns
        -------
        The matplotlib Axes object.
        """
        self.add_title(title)
        linespaces = self._create_linespaces(offset, data_points)
        self.__plot_functions(linespaces)
        if cartesian_plane:
            self.__create_cartesian_plane()
        self._axes.legend()
        return self._axes

    # endregion

    # region ITERATIONS
    def _plot_iteration(self, i: int, iteration: Any) -> None:
        """
        Override in subclasses to draw artists for one iteration.

        Every artist appended to self._iteration_artists will be removed
        automatically before the next frame is rendered.

        Parameters
        ----------
        i         : Zero-based frame / row index.
        iteration : The corresponding row as a pandas Series (from .iloc).
        """
        pass

    def plot_iterations(self, iterations: pd.DataFrame | None = None) -> None:
        """
        Draw all iterations statically (no animation).

        Parameters
        ----------
        iterations : DataFrame to iterate; defaults to self._iterations.
        """
        data = iterations if iterations is not None else self._iterations
        for i, row in enumerate(data.itertuples(index=False)):
            # Convert namedtuple row to a Series so subclasses always
            # receive the same type regardless of how the method is called.
            series = pd.Series(row._asdict())
            self._plot_iteration(i, series)

    # endregion

    # region ANIMATION
    def _clear_iteration_artists(self) -> None:
        """Remove all artists that were added during the previous frame."""
        for artist in self._iteration_artists:
            artist.remove()
        self._iteration_artists.clear()

    def _update(self, frame: int) -> Axes:
        """FuncAnimation callback: clear previous frame, draw the new one."""
        self._clear_iteration_artists()
        iteration = self._iterations.iloc[frame]
        self._plot_iteration(frame, iteration)
        return self._axes

    def animate_iterations(
        self,
        title: str,
        interval: int = 400,
        repeat: bool = True,
    ) -> animation.FuncAnimation:
        """
        Animate self._iterations frame-by-frame on top of the base graph.

        Parameters
        ----------
        title    : Chart title passed to create_graph().
        interval : Delay between frames in milliseconds.
        repeat   : Whether the animation loops.

        Returns
        -------
        The FuncAnimation object (useful for saving with .save()).
        """
        n_frames = len(self._iterations)
        if n_frames == 0:
            raise RuntimeError("No iteration data to animate.")

        self.create_graph(title)

        self._ani = animation.FuncAnimation(
            self._fig,
            self._update,
            frames=n_frames,
            interval=interval,
            blit=False,
            repeat=repeat,
        )
        return self._ani

    # endregion

    # region SHOW / CLEAR
    def show(self) -> None:
        """Render the figure (tight layout, then plt.show)."""
        self._fig.tight_layout()
        plt.show()

    def clear(self) -> None:
        """Clear the axes without destroying the figure."""
        self._axes.cla()

    # endregion

# endregion


# ---------------------------------------------------------------------------
# Minimal usage example / smoke test
# ---------------------------------------------------------------------------
def test() -> None:
    """
    Demonstrate a concrete subclass that visualizes Newton's method
    converging on the root of f(x) = x^2 - 2.
    """

    class NewtonVisualizer(Visualizer):
        def _find_range(self) -> None:
            if not self._iterations.empty:
                self.begin = float(self._iterations["x"].min()) - 1
                self.end = float(self._iterations["x"].max()) + 1
            else:
                self.begin, self.end = -3.0, 3.0

        def _plot_iteration(self, i: int, iteration: Any) -> None:
            x = float(iteration["x"])
            y = float(iteration["y"])

            # Vertical line from (x, y) down to x-axis
            vline = self._axes.plot(
                [x, x], [0, y], color="red", linestyle="--", linewidth=0.8, alpha=0.6
            )[0]
            dot = self._axes.plot(x, y, "ro", markersize=5)[0]

            self._iteration_artists.extend([vline, dot])

    # Build Newton's-method iteration table for f(x) = x^2 - 2
    def f(x):  return x**2 - 2
    def df(x): return 2 * x

    rows, x = [], 3.0
    for _ in range(8):
        rows.append({"x": x, "y": f(x)})
        x = x - f(x) / df(x)

    iters = pd.DataFrame(rows)

    vis = NewtonVisualizer(iters, [f])
    vis.plot_iterations()          # static view
    vis.create_graph("Newton's method  —  f(x) = x² − 2")
    vis.show()


def test_animation() -> None:
    """
    Verify that animate_iterations() works end-to-end through NewtonVisualizer.

    What this checks
    ----------------
    1. NewtonVisualizer (a Visualizer subclass) can be instantiated.
    2. _find_range()      is called on __init__ and sets begin / end correctly.
    3. animate_iterations() builds a FuncAnimation without raising.
    4. _update()          is callable for every frame index (0 … n-1).
    5. _clear_iteration_artists() removes artists between frames without error.
    6. The returned object is a FuncAnimation instance.
    7. The animation covers exactly as many frames as there are iteration rows.

    The test runs headlessly (matplotlib's 'Agg' backend) so it works in
    environments without a display.  Switch back to the interactive backend
    after the test so subsequent plt.show() calls behave normally.
    """
    import matplotlib
    _original_backend = matplotlib.get_backend()
    matplotlib.use("Agg")           # headless — no window needed

    # ------------------------------------------------------------------
    # 1. Define the same NewtonVisualizer used in test()
    # ------------------------------------------------------------------
    class NewtonVisualizer(Visualizer):
        def _find_range(self) -> None:
            if not self._iterations.empty:
                self.begin = float(self._iterations["x"].min()) - 1
                self.end   = float(self._iterations["x"].max()) + 1
            else:
                self.begin, self.end = -3.0, 3.0

        def _plot_iteration(self, i: int, iteration: Any) -> None:
            x = float(iteration["x"])
            y = float(iteration["y"])
            vline = self._axes.plot(
                [x, x], [0, y], color="red", linestyle="--", linewidth=0.8, alpha=0.6
            )[0]
            dot = self._axes.plot(x, y, "ro", markersize=5)[0]
            self._iteration_artists.extend([vline, dot])

    # ------------------------------------------------------------------
    # 2. Build iteration data
    # ------------------------------------------------------------------
    def f(x):  return x**2 - 2
    def df(x): return 2 * x

    rows, x = [], 3.0
    for _ in range(8):
        rows.append({"x": x, "y": f(x)})
        x = x - f(x) / df(x)

    iters = pd.DataFrame(rows)
    n_frames = len(iters)

    # ------------------------------------------------------------------
    # 3. Instantiate and verify _find_range ran
    # ------------------------------------------------------------------
    vis = NewtonVisualizer(iters, [f])

    assert vis.begin < vis.end, (
        f"_find_range() did not set a valid range: begin={vis.begin}, end={vis.end}"
    )
    print(f"  [PASS] _find_range  →  begin={vis.begin:.4f}, end={vis.end:.4f}")

    # ------------------------------------------------------------------
    # 4. Run animate_iterations (returns FuncAnimation, does not plt.show
    #    because Agg is headless)
    # ------------------------------------------------------------------
    ani = vis.animate_iterations(
        title="Animation test — Newton's method on f(x) = x² − 2",
        interval=200,
        repeat=False,
    )

    assert isinstance(ani, animation.FuncAnimation), (
        f"animate_iterations() should return FuncAnimation, got {type(ani)}"
    )
    print(f"  [PASS] animate_iterations  →  returned FuncAnimation")

    # ------------------------------------------------------------------
    # 5. Manually drive every frame through _update and verify artists
    #    are created then cleared correctly
    # ------------------------------------------------------------------
    for frame in range(n_frames):
        vis._update(frame)

        artists_after_update = len(vis._iteration_artists)
        assert artists_after_update > 0, (
            f"Frame {frame}: _plot_iteration() added no artists to "
            f"_iteration_artists — did you forget self._iteration_artists.extend(...) ?"
        )

        vis._clear_iteration_artists()
        assert len(vis._iteration_artists) == 0, (
            f"Frame {frame}: _clear_iteration_artists() left "
            f"{len(vis._iteration_artists)} artist(s) behind."
        )

    print(f"  [PASS] _update / _clear_iteration_artists  →  {n_frames} frames OK")

    # ------------------------------------------------------------------
    # 6. Verify frame count matches iteration rows
    # ------------------------------------------------------------------
    actual_frames = getattr(ani, "_save_count", None)
    assert actual_frames == n_frames, (
        f"Frame count mismatch: FuncAnimation has {actual_frames} frames, "
        f"expected {n_frames}."
    )
    print(f"  [PASS] frame count  →  {actual_frames} frames match iteration rows")

    # ------------------------------------------------------------------
    # 7. Restore original backend
    # ------------------------------------------------------------------
    plt.close("all")
    matplotlib.use(_original_backend)

    print("\n  All animation tests passed.")


if __name__ == "__main__":
    # print("Running test() ...")
    # test()
    print("\nRunning test_animation() ...")
    test_animation()