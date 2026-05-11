from ..Numerix import Numerix
from typing import Callable
import inspect


class Sle(Numerix):
    def __init__(self, is_verbose=False):
        super().__init__(is_verbose=is_verbose)
        self.functions: list[Callable] = []
        self._argument_count: int | None = None

    # region ARGUMENTS
    @property
    def functions(self):
        return self._functions

    @functions.setter
    def functions(self, funcs):
        self._functions = []
        self._argument_count = None

        for func in funcs:
            self.add_function(func)

    def __check_arg_count(self, function: Callable):
        signature = inspect.signature(function)
        arg_count = len(signature.parameters)
        if self._argument_count is None:
            self._argument_count = arg_count
        if arg_count != self._argument_count:
            raise ValueError(
                "Function argument count mismatch. "
                f"Expected {self._argument_count}, "
                f"got {arg_count}."
            )

    def _validate_function(self, function: Callable, check_arg=True):
        self._check_callable(function)
        if check_arg:
            self.__check_arg_count(function)

    def add_function(self, function: Callable):
        """
        Store mathematical function and enforce
        consistent argument count.
        """
        self._validate_function(function)
        self.functions.append(function)
        if self._is_verbose:
            print(f"Function added with " f"{self._argument_count} argument(s).")

    def create_first_arguments(self):
        return (0,) * self._argument_count

    # endregion

    # region ITERATION
    def create_iteration(self, itr, args, old_args=None):
        iteration = {"iteration": itr, "args": args}
        for func in self.functions:
            iteration[func.__name__] = func(*args)
        if old_args is not None:
            ea = self.calculate_ea(args, old_args)
            iteration["ea"] = ea
            iteration["|ea|"] = max(abs(i) for i in ea)
            iteration["|ea|%"] = iteration["|ea|"] * 100
        else:
            iteration["ea"] = None
            iteration["|ea|"] = None
            iteration["|ea|%"] = None
        return iteration

    def calculate_ea(self, x_new, x_old):
        errors = []
        for new, old in zip(x_new, x_old):
            if new != 0:
                errors.append((new - old) / new)
            else:
                errors.append(new - old)
        return errors
    # endregion
    
    #region CALCULATE
    def _compute_next_arguments(self, old_args):
        # to be overridden
        pass

    def calculate(self, ea_tol=1e-6, max_iter=100):
        if self._argument_count != len(self.functions):
            raise ValueError(
                f"Arguments count must be the same as the number of functions.\n"
                + f"Arguments count: {self._argument_count}\n"
                + f"Functions count: {len(self.functions)}"
            )
        self.iterations = []

        args = self.create_first_arguments()
        self.iterations.append(self.create_iteration(0,args))

        for i in range(max_iter):
            old_args = args
            args = self._compute_next_arguments(old_args)
            iteration = self.create_iteration(i+1,args, old_args)
            self.iterations.append(iteration)
            ea = iteration["|ea|"]
            residuals = [abs(iteration[func.__name__]) for func in self.functions]
            max_residual = max(residuals)
            if ea is not None and ea < ea_tol:
                return args
        Warning(f"Method did not converge within {max_iter} iterations")
        return args
    #endregion
