from ..Numerix import Numerix


class Root(Numerix):
    def __init__(self, is_verbose=False):
        super().__init__(is_verbose)

    @property
    def function(self):
        return self._function

    @function.setter
    def function(self, value):
        self._validate_function(value,check_arg=False)
        self._function = value
