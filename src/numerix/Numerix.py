import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import inspect

from typing import Dict, Any, Callable


class Numerix:
    """
    Base class for numerical method implementations.
    """

    def __init__(self, is_verbose: bool = False):

        self.__is_verbose = is_verbose

        self.__columns: list[str] = []
        self.__linespaces: list[tuple[np.ndarray, np.ndarray]] = []

        self.__min_x = None
        self.__max_x = None

        self.initiator = None
        self.iterations = pd.DataFrame()

        self.functions: list[Callable] = []
        self.argument_count: int | None = None

    # region Properties
    def __challenge_min_max_x(self, iteration):
        pass

    def _process_on_each_iterations(self, iteration: Dict[str, Any]):
        # to be overriden
        pass

    def add_iterations(self, iteration: Dict[str, Any]):
        if not hasattr(self, "__columns"):
            self.__columns = list(iteration.keys())
            self.iterations = pd.DataFrame(columns=self.__columns)

        if list(iteration.keys()) != self.__columns:
            raise ValueError("Iteration columns do not match previous entries.")

        self.iterations.loc[len(self.iterations)] = iteration

        self.__find_min_max_x(iteration)

        if self.__is_verbose:
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

        if self.__is_verbose:
            print(f"Function added with " f"{arg_count} argument(s).")

    # region Visualization
    def display_iterations(self):
        if self.iterations.empty:
            print("No iterations recorded.")
            return

        print(self.iterations)

    def __create_linespace(self, function: Callable):
        if self.__min_x is None or self.__max_x is None:
            return

        x = np.linspace(self.__min_x, self.__max_x)
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

    def plot_iterations(self, **plot_kwargs):
        if not self.functions:
            raise RuntimeError("No functions added.")

        self.__create_linespaces_for_all()
        self.__create_plot_from_linespace()

        plt.axhline(0)
        plt.axvline(0)

        plt.grid(True)

        if plot_kwargs.get("legend", True):
            plt.legend()

        plt.show()
