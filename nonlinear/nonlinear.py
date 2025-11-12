# ================================================================
# NONLINEAR SYSTEM BASE CLASS (nonlinear_system.py)
# ================================================================
class NonLinearSystem(Numerix):
    """
    Base class for systems of nonlinear equations (SNLE).

    Handles:
    - Vector-valued equation systems F(x) = [f1, f2, ..., fn]
    - Initial guesses
    - Precision and verbosity inherited from Numerix
    """
    def __init__(self, base_equation, initial_guess):
        super().__init__(base_equation)
        self.set_initial_guess(initial_guess)
        self._verbose = is_verbose()
        self._dtype = dtype()
        if self._verbose:
            print(f"[NonLinearSystem] Initialized with initial guess: {self.initial_guess}")

    # -----------------------------------------------------------
    # Override base equation handling for vector systems
    # -----------------------------------------------------------
    def set_base_equation(self, base_equation):
        """
        Override Numerix.set_base_equation to handle systems (lists/tuples of callables).
        """
        if isinstance(base_equation, (list, tuple)):
            # Validate that all elements are callable
            if not all(callable(eq) for eq in base_equation):
                raise TypeError("All elements of base_equation must be callable functions.")
            self._base_equation = base_equation
            if self._verbose:
                print(f"[NonLinearSystem] Registered system of {len(base_equation)} equations.")
        else:
            # Fallback to Numerix single-equation handler
            super().set_base_equation(base_equation)

    # -----------------------------------------------------------
    # Initial Guess Handling
    # -----------------------------------------------------------
    def set_initial_guess(self, initial_guess):
        arr = np.array(initial_guess, dtype=dtype())
        if arr.ndim != 1:
            raise ValueError("Initial guess must be a 1D vector.")
        self.initial_guess = arr
        if self._verbose:
            print(f"[Numerix] Initial guess set to: {self.initial_guess}")

    def get_initial_guess(self) -> np.ndarray:
        return self.initial_guess
