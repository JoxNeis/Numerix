from ..Sle import Sle


class Jacobi(Sle):
    def __init__(self, is_verbose=False):
        super().__init__(is_verbose)

    def _compute_next_arguments(self, old_args):
        """
        Compute the next approximation using the Jacobi iteration formula.

        For a system:
            f1(x1, x2, ..., xn) = 0
            f2(x1, x2, ..., xn) = 0
            ...
            fn(x1, x2, ..., xn) = 0

        Each function in self.functions must be arranged so that it returns the
        isolated variable value:
            x1 = g1(x1, x2, ..., xn)
            x2 = g2(x1, x2, ..., xn)
            ...
            xn = gn(x1, x2, ..., xn)

        Jacobi uses ONLY the previous iteration values (old_args) to compute
        all new values.
        """
        new_args = tuple(func(*old_args) for func in self.functions)

        return new_args

    def calculate(self, ea_tol=1e-6, max_iter=200):
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


if __name__ == "__main__":
    # 10x - y + 2z = 6
    # -x + 11y - z = 25
    # 2x - y + 10z = -11
    #
    # Rearranged into Jacobi form:
    # x = (6 + y - 2z) / 10
    # y = (25 + x + z) / 11
    # z = (-11 - 2x + y) / 10

    def g1(x, y, z):
        return (6 + y - 2 * z) / 10

    def g2(x, y, z):
        return (25 + x + z) / 11

    def g3(x, y, z):
        return (-11 - 2 * x + y) / 10

    jacobi = Jacobi()
    jacobi.add_function(g1)
    jacobi.add_function(g2)
    jacobi.add_function(g3)

    solution = jacobi.calculate(
        ea_tol=1e-6
    )

    print("\nSolution found:")
    print(f"x = {solution[0]:.6f}")
    print(f"y = {solution[1]:.6f}")
    print(f"z = {solution[2]:.6f}")

    print("\nVerification:")
    x, y, z = solution
    print(f"10x - y + 2z = {10*x - y + 2*z:.6f} (should be 6)")
    print(f"-x + 11y - z = {-x + 11*y - z:.6f} (should be 25)")
    print(f"2x - y + 10z = {2*x - y + 10*z:.6f} (should be -11)")

    print("\nIterations:")
    from tabulate import tabulate
    print(tabulate(jacobi.get_iterations(), headers='keys', tablefmt='psql'))
    