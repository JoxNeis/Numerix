import types
import pandas as pd
from typing import Any, Dict, Callable

class Numerix:
    """Base class for all numerical methods in the Numerix library.

    Provides:
    - Validation for base equations (must be lambda functions)
    - A process log (pandas DataFrame) for iterative tracking
    """

    def __init__(self, base_equation: Callable):
        self.set_base_equation(base_equation)
        self.set_process_log()

    def set_base_equation(self, base_equation: Callable):
        """Set the base equation for the numerical method."""
        if self._is_lambda_function(base_equation):
            self._base_equation = base_equation
        else:
            raise ValueError("All equations must be lambda functions (anonymous).")

    def get_base_equation(self) -> Callable:
        """Return the stored base equation."""
        return self._base_equation

    def set_process_log(self):
        """Initialize the process log as an empty pandas DataFrame."""
        self._process_log = pd.DataFrame()

    def get_process_log(self) -> pd.DataFrame:
        """Return the process log DataFrame."""
        return self._process_log

    def append_process_log(self, log: Dict[str, Any]):
        """Append a new record to the process log safely."""
        if not isinstance(log, dict):
            raise TypeError("Process log entry must be a dictionary.")
        # Using concat instead of deprecated .append()
        self._process_log = pd.concat(
            [self._process_log, pd.DataFrame([log])],
            ignore_index=True
        )

    @staticmethod
    def _is_lambda_function(obj: Any) -> bool:
        """Check whether the given object is a lambda function."""
        return isinstance(obj, types.LambdaType) and obj.__name__ == "<lambda>"
