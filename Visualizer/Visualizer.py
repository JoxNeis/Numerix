import matplotlib.pyplot as plt
import numpy as np
from typing import Callable


class Visualizer:
    def __init__(self):
        self._fig, self._axes = plt.subplots()

    # region GRAPH GENERATION
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
    def __create_cartesian_plane(
        self,
        axis_color="black",
        axis_width=1,
        grid_style="--",
        grid_width=0.5,
        grid_transparency=0.4,
    ) -> None:
        self._axes.axhline(0, color=axis_color, linewidth=axis_width,label='X')
        self._axes.axvline(0, color=axis_color, linewidth=axis_width,label='Y')

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

    def __get_function_name(
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

            name = self._get_function_name(func, i)

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
    
    #endregion
    
    #region ANIMATION
    
    #endregion
    
    #region SHOW
    def show(self) -> None:
        self._fig.tight_layout()
        plt.show()

    def clear(self) -> None:
        self._axes.cla()
    #endregion
