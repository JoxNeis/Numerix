from ..Root import Root


class Bracketing(Root):
    def __init__(self, is_verbose=False):
        super().__init__(is_verbose)

    def _check_brackets(self, low, upper):
        f_low = self.function(low)
        f_upper = self.function(upper)

        if (f_low * f_upper) >= 0:
            raise ValueError(
                f"Invalid bracketing interval: f({low}) = {f_low}, "
                f"f({upper}) = {f_upper}. "
                "Function values at the interval endpoints must have opposite signs "
                "(i.e., f(low) * f(upper) < 0)."
            )

