import numpy as np

class Binary:
    
    @staticmethod
    def int2bin(n : int, bits : int):
        return f"{n:0{bits}b}"
    
    @staticmethod
    def popcount(n : int):
        return n.bit_count()
    
    