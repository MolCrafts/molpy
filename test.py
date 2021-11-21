from copy import deepcopy

class A:
    
    def __init__(self) -> None:
        self.data = {'key': 'value'}
        self.stable = 'stable'
        self.myself = self
        
a = A()
aa = deepcopy(a)