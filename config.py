import warnings
from typing import Literal

# ================================================================
# CONFIGURATION SYSTEM (config.py)
# ================================================================
_DEFAULT_PRECISION: Literal['float32', 'float64'] = 'float64'
_DEFAULT_BACKEND: Literal['numpy', 'numba'] = 'numpy'
_VERBOSE: bool = False
_ABS_TOL: float = 1e-8
_REL_TOL: float = 1e-6

# -------------------- GETTERS --------------------
def get_precision() -> str:
    return _DEFAULT_PRECISION

def get_backend() -> str:
    return _DEFAULT_BACKEND

def is_verbose() -> bool:
    return _VERBOSE

def get_tolerances() -> tuple[float, float]:
    return _ABS_TOL, _REL_TOL

# -------------------- SETTERS --------------------
def set_precision(precision: Literal['float32', 'float64']) -> None:
    global _DEFAULT_PRECISION
    if precision not in ('float32', 'float64'):
        raise ValueError("Precision must be 'float32' or 'float64'.")
    _DEFAULT_PRECISION = precision
    if _VERBOSE:
        print(f"[Numerix] Precision set to: {_DEFAULT_PRECISION}")

def set_backend(backend: Literal['numpy', 'numba']) -> None:
    global _DEFAULT_BACKEND
    if backend not in ('numpy', 'numba'):
        raise ValueError("Backend must be 'numpy' or 'numba'.")
    _DEFAULT_BACKEND = backend
    if _VERBOSE:
        print(f"[Numerix] Backend set to: {_DEFAULT_BACKEND}")

def set_verbose(flag: bool) -> None:
    global _VERBOSE
    _VERBOSE = bool(flag)
    print(f"[Numerix] Verbose mode {'enabled' if _VERBOSE else 'disabled'}.")

def set_tolerances(abs_tol: float, rel_tol: float) -> None:
    global _ABS_TOL, _REL_TOL
    if abs_tol <= 0 or rel_tol <= 0:
        raise ValueError("Tolerances must be positive.")
    _ABS_TOL, _REL_TOL = abs_tol, rel_tol
    if _VERBOSE:
        print(f"[Numerix] Tolerances set to abs={_ABS_TOL:.2e}, rel={_REL_TOL:.2e}")

# -------------------- UTILITIES --------------------
def dtype() -> np.dtype:
    return np.float32 if _DEFAULT_PRECISION == 'float32' else np.float64

def summary() -> None:
    print("───────────────────────────────")
    print(" Numerix Runtime Configuration")
    print("───────────────────────────────")
    print(f" Precision     : {_DEFAULT_PRECISION}")
    print(f" Backend       : {_DEFAULT_BACKEND}")
    print(f" Verbose       : {_VERBOSE}")
    print(f" Abs Tolerance : {_ABS_TOL:.2e}")
    print(f" Rel Tolerance : {_REL_TOL:.2e}")
    print("───────────────────────────────")

def validate_environment() -> None:
    if _DEFAULT_BACKEND == 'numba':
        try:
            import numba  # noqa
        except ImportError:
            warnings.warn("[Numerix] Numba backend selected but not installed. Falling back to NumPy.")
            set_backend('numpy')
    if _VERBOSE:
        print("[Numerix] Environment validated successfully.")

