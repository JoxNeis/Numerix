class RegulaFalsi(Bracketing):
    def calculate(self, tolerance: float = 1e-4, max_iterations: int = 100):

        if not self.boundaries:
            raise RuntimeError("Initial boundary not set.")

        lower = self.boundaries[0]["lower"]
        upper = self.boundaries[0]["upper"]

        previous_midpoint = None

        for iteration in range(max_iterations):

            f_low = self.function(lower)
            f_upper = self.function(upper)

            midpoint = (
                f_upper * lower
                - f_low * upper
            ) / (f_upper - f_low)

            f_mid = self.function(midpoint)

            if previous_midpoint is None:
                ea = None
                er = None
            else:
                ea = abs(midpoint - previous_midpoint)
                er = ea / abs(midpoint) if midpoint != 0 else None

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

            if abs(f_mid) < tolerance:
                return midpoint

            if ea is not None and ea < tolerance:
                return midpoint

            if f_low * f_mid < 0:
                upper = midpoint
            else:
                lower = midpoint

            previous_midpoint = midpoint

        return midpoint