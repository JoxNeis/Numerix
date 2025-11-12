
# ================================================================
# NEWTON-RAPHSON SOLVER (newton.py)
# ================================================================
class NewtonRaph(NonLinearSystem):
    def __init__(self, base_equations, derivatives, initial_guess):
        super().__init__(base_equations, initial_guess)
        self.set_derivatives(derivatives)
        if self._verbose:
            print(f"[Newton] Initialized with {len(base_equations)} equations and Jacobian matrix of shape {self._jacobian_shape}.")

    def set_derivatives(self, derivatives):
        if not isinstance(derivatives, (list, tuple)):
            raise TypeError("Derivatives must be a list (of lists) of callable partial derivatives.")
        for row in derivatives:
            if not all(callable(d) for d in row):
                raise TypeError("Every derivative must be callable.")
        self._derivatives = derivatives
        self._jacobian_shape = (len(derivatives), len(derivatives[0]))

    def evaluate_jacobian(self, args):
        return np.array([
            [der(*args) for der in row]
            for row in self._derivatives
        ], dtype=dtype())

    def solve(self, max_iter=100, tol=_ABS_TOL):
        if self._verbose:
            print(f"[Newton] Starting iterative solution (max_iter={max_iter}, tol={tol:.1e})")

        prev_args = np.array(self.initial_guess, dtype=dtype())

        for i in range(max_iter):
            Fx = np.array([f(*prev_args) for f in self._base_equation], dtype=dtype())
            Jx = self.evaluate_jacobian(prev_args)

            if np.linalg.det(Jx) == 0:
                raise np.linalg.LinAlgError(f"Jacobian is singular at iteration {i+1}.")

            inv_J = np.linalg.inv(Jx)
            delta = inv_J @ Fx
            new_args = prev_args - delta

            self.append_process_log({
                "Iteration": i + 1,
                "Args": prev_args.copy(),
                "F(x)": Fx.copy(),
                "Δx": delta.copy(),
                "‖Δx‖₂": np.linalg.norm(delta),
                "New Args": new_args.copy(),
            })

            if np.linalg.norm(delta) < tol:
                if self._verbose:
                    print(f"[Newton Raphson] Converged after {i+1} iterations.")
                return new_args, self.get_process_log()

            prev_args = new_args

        if self._verbose:
            print("[Newton Raphson] Maximum iterations reached without convergence.")
        return prev_args, self.get_process_log()
