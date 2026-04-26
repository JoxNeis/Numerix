import pandas as pd
import numpy as np
import inspect

from typing import Callable, Dict, Any


# region NUMERIX
class Numerix:
    """
    Base class for numerical method implementations.
    """

    def __init__(self, is_verbose: bool = False):

        self._is_verbose = is_verbose
        self.iterations: list[Dict[str, Any]] = []
        self.functions: list[Callable] = []
        self.argument_count: int | None = None

    # region Properties

    def _after_each_iteration(self, iteration: Dict[str, Any]):
        # to be overriden
        pass

    def add_iterations(self, iteration: Dict[str, Any]):
        self.iterations.append(iteration)
        self._after_each_iteration(iteration)
        if self._is_verbose:
            print(iteration)

    def __check_callable(self, function: Callable):
        if not callable(function):
            raise TypeError("Function must be callable.")
        
    def __check_arg_count(self,function:Callable):
        signature = inspect.signature(function)
        arg_count = len(signature.parameters)
        if self.argument_count is None:
            self.argument_count = arg_count
        if arg_count != self.argument_count:
            raise ValueError(
                "Function argument count mismatch. "
                f"Expected {self.argument_count}, "
                f"got {arg_count}."
            )
        

    def _validate_function(self, function: Callable):
        self.__check_callable(function)
        self.__check_arg_count(function)
        

    def add_function(self, function: Callable):
        """
        Store mathematical function and enforce
        consistent argument count.
        """
        self._validate_function(function)
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


# endregion
