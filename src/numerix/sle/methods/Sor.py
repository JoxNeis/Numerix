from ..Sle import Sle


class Sor(Sle):
    def __init__(self, omega=1.25, is_verbose=False):
        super().__init__(is_verbose)
        self.omega = omega

    def _compute_next_arguments(self, old_args):
        args = list(old_args)
        for i, func in enumerate(self.functions):
            gs = func(*args)
            args[i] = self.omega * gs + (1 - self.omega) * old_args[i]
        return tuple(args)

if __name__ == "__main__":
    # 10x - y + 2z = 6
    # -x + 11y - z = 25
    # 2x - y + 10z = -11
    #
    # Rearranged into sor form:
    # x = (6 + y - 2z) / 10
    # y = (25 + x + z) / 11
    # z = (-11 - 2x + y) / 10

    def g1(x, y, z):
        return (6 + y - 2 * z) / 10

    def g2(x, y, z):
        return (25 + x + z) / 11

    def g3(x, y, z):
        return (-11 - 2 * x + y) / 10

    sor = Sor(1.25)
    sor.add_function(g1)
    sor.add_function(g2)
    sor.add_function(g3)

    solution = sor.calculate(
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
    print(sor.get_iterations())
    