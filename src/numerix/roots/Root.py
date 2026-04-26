from ..Numerix import Numerix

class Root(Numerix):
    def __init__(self,is_verbose = False):
        super(self,is_verbose)
        
    def add_function(self, function):
        self.argument_count = None
        self.functions.clear()
        super().add_function(function)
        self.function = self.functions[0]
        
    
    