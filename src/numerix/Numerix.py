import pandas as pd
from typing import Dict, Any

class Numerix:
    """
    Base class for numerical method implementations.

    This class provides common configuration utilities such as verbosity
    control and structured logging. It is intended to be inherited by
    concrete numerical method classes.

    Attributes
    ----------
    is_verbose : bool
        If True, enables verbose output for debugging or progress reporting.
    is_logging : bool
        If True, enables logging of runtime information into a pandas DataFrame.
    logs : pd.DataFrame
        DataFrame storing log entries. Initialized empty and populated via
        `add_logs` when logging is enabled.
    _log_columns : list[str]
        Internal list defining the expected log schema (column names).
        Determined by the first log entry and enforced for all subsequent logs.

    Examples
    --------
    Basic usage with logging enabled:

    >>> nx = Numerix(is_logging=True)
    >>> nx.add_logs({"iter": 1, "loss": 0.5})
    >>> nx.add_logs({"iter": 2, "loss": 0.25})
    >>> list(nx.logs.columns)
    ['iter', 'loss']
    >>> len(nx.logs)
    2

    Logging disabled raises an error:

    >>> nx = Numerix(is_logging=False)
    >>> nx.add_logs({"iter": 1})
    Traceback (most recent call last):
    ...
    RuntimeError: is_logging is set to False. Logging is not possible.

    Inconsistent log schema raises an error:

    >>> nx = Numerix(is_logging=True)
    >>> nx.add_logs({"iter": 1, "loss": 0.5})
    >>> nx.add_logs({"iteration": 2, "loss": 0.25})
    Traceback (most recent call last):
    ...
    ValueError: Log keys do not match previous logs. Ensure consistent log structure for every entry.
    """

    def __init__(self, is_verbose: bool = False, is_logging: bool = False):
        """
        Initialize the Numerix base class.

        Parameters
        ----------
        is_verbose : bool, optional
            Enables verbose output if set to True.
        is_logging : bool, optional
            Enables logging functionality if set to True.

        Examples
        --------
        >>> nx = Numerix()
        >>> nx.is_verbose
        False
        >>> nx.is_logging
        False
        """
        self.is_verbose = is_verbose
        self.is_logging = is_logging
        self.logs = pd.DataFrame()
        self._log_columns: list[str] = []

    def add_logs(self, log: Dict[str, Any]):
        """
        Add a log entry to the internal log DataFrame.

        The first call initializes the logging schema using the keys of
        the provided dictionary. All subsequent calls must provide
        dictionaries with identical keys and order.

        Parameters
        ----------
        log : Dict[str, Any]
            A dictionary representing a single log entry.

        Raises
        ------
        RuntimeError
            If logging is disabled.
        ValueError
            If the log dictionary keys do not match the existing schema.

        Examples
        --------
        >>> nx = Numerix(is_logging=True)
        >>> nx.add_logs({"step": 1, "value": 10})
        >>> nx.add_logs({"step": 2, "value": 8})
        >>> nx.logs.iloc[0]["value"]
        10

        >>> nx.add_logs({"step": 3, "val": 5})
        Traceback (most recent call last):
        ...
        ValueError: Log keys do not match previous logs. Ensure consistent log structure for every entry.
        """
        if not self.is_logging:
            raise RuntimeError(
                "is_logging is set to False. Logging is not possible."
            )

        if not self._log_columns:
            self._log_columns = list(log.keys())
            self.logs = pd.DataFrame(columns=self._log_columns)
        else:
            if list(log.keys()) != self._log_columns:
                raise ValueError(
                    "Log keys do not match previous logs. "
                    "Ensure consistent log structure for every entry."
                )

        self.logs.loc[len(self.logs)] = log
