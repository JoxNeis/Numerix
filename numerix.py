import numpy as np
import pandas as pd
import warnings
import types
import numerix.config
from typing import Any, Dict, Callable, Literal

# ================================================================
# NUMERIX BASE CLASS (numerix.py)
# ================================================================
class Numerix:
    def __init__(self, base_equation: Callable):
        self._dtype = dtype()
        self._abs_tol, self._rel_tol = get_tolerances()
        self._backend = get_backend()
        self._verbose = is_verbose()

        self.set_base_equation(base_equation)
        self.set_process_log()

        if self._verbose:
            print(f"[Numerix] Initialized with dtype={self._dtype}, backend={self._backend}")

    # Base Equation
    def set_base_equation(self, base_equation: Callable):
        if self._is_lambda_function(base_equation):
            self._base_equation = base_equation
        elif callable(base_equation):
            if self._verbose:
                print("[Numerix] Warning: Base equation is callable but not a lambda.")
            self._base_equation = base_equation
        else:
            raise TypeError("Base equation must be a lambda or callable function.")

    def get_base_equation(self) -> Callable:
        return self._base_equation

    # Process Log
    def set_process_log(self):
        self._process_log = pd.DataFrame(dtype=self._dtype)
        if self._verbose:
            print("[Numerix] Process log initialized.")

    def append_process_log(self, log: Dict[str, Any]):
        if not isinstance(log, dict):
            raise TypeError("Process log entry must be a dictionary.")
        for k, v in log.items():
            if isinstance(v, (int, float)):
                log[k] = self._dtype(v)
        self._process_log = pd.concat([self._process_log, pd.DataFrame([log])], ignore_index=True)
        if self._verbose:
            print(f"[Numerix] Appended log entry: {log}")

    def get_process_log(self) -> pd.DataFrame:
        return self._process_log

    # Config and Summary
    def get_config(self) -> Dict[str, Any]:
        return {
            "dtype": self._dtype,
            "abs_tol": self._abs_tol,
            "rel_tol": self._rel_tol,
            "backend": self._backend,
            "verbose": self._verbose,
        }

    def summary(self):
        print("───────────────────────────────")
        print(" Numerix Instance Configuration")
        print("───────────────────────────────")
        print(f" Precision (dtype) : {self._dtype}")
        print(f" Backend           : {self._backend}")
        print(f" Abs Tol           : {self._abs_tol:.2e}")
        print(f" Rel Tol           : {self._rel_tol:.2e}")
        print(f" Verbose           : {self._verbose}")
        print("───────────────────────────────")

    @staticmethod
    def _is_lambda_function(obj: Any) -> bool:
        return isinstance(obj, types.LambdaType) and obj.__name__ == "<lambda>"