import Bracketing

class Bisection(Bracketing):
    def calculate(self, tolerance: float = 1e-4, max_iterations: int = 100):
        if not self.boundaries:
            raise RuntimeError("Initial boundary not set.")

        lower = self.boundaries[0]["lower"]
        upper = self.boundaries[0]["upper"]

        f_low = self.function(lower)
        previous_midpoint = None

        for iteration in range(max_iterations):
            midpoint = (lower + upper) / 2
            f_mid = self.function(midpoint)

            ea = (
                abs(midpoint - previous_midpoint)
                if previous_midpoint is not None
                else None
            )
            er = (ea / abs(midpoint)) if (ea is not None and midpoint != 0) else None

            self.add_iterations(
                {
                    "iteration": iteration,
                    "lower": lower,
                    "upper": upper,
                    "midpoint": midpoint,
                    "f_mid": f_mid,
                    "ea": ea,
                    "er": er,
                    "Ea": "Unknown",
                    "Er": "Unknown",
                }
            )

            if f_mid == 0 or (ea is not None and ea < tolerance):
                if self._is_verbose:
                    print("Iteration stopped, reached tolerance")
                return midpoint

            if f_low * f_mid < 0:
                upper = midpoint
            else:
                lower = midpoint
                f_low = f_mid

            previous_midpoint = midpoint

        return midpoint
