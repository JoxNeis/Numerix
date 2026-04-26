from ..Root import Root


class Bracketing(Root):
    def __init__(self, is_verbose=False):
        super(self, is_verbose)

    def _check_brackets(self, low, upper):
        return (self.function(low) * self.function(upper)) < 0
