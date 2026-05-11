from ..Numerix import Numerix
from typing import Callable


class Root(Numerix):
    def __init__(self, is_verbose=False):
        super().__init__(is_verbose)

    @property
    def function(self):
        return self._function

    @function.setter
    def function(self, value:Callable):
        self._validate_function(value,check_arg=False)
        self._function = value
