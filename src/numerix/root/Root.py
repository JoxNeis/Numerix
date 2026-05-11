from ..Numerix import Numerix

class Root(Numerix):
    def __init__(self,is_verbose = False):
        super().__init__(is_verbose)
        
    def add_function(self, function):
        self.argument_count = None
        self.functions.clear()
        super().add_function(function)
        
    @property
    def function(self):
        return self.functions[0]

        
    
    