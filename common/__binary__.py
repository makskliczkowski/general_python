import numpy as np

class Binary:
    """
    A class containing static methods for binary operations. 
    """
    
    @staticmethod
    def int2bin(n : int, bits : int):
        """
        Convert an integer to its binary representation with a fixed number of bits.

        Args:
            n (int): The integer to convert.
            bits (int): The number of bits in the binary representation.

        Returns:
            str: The binary representation of the integer, padded with leading zeros to fit the specified number of bits.
        """
        return f"{n:0{bits}b}"
    
    ####################################################################################################
    
    @staticmethod
    def popcount(n : int):
        """
        Calculate the number of 1-bits in the binary representation of an integer.

        Args:
            n (int): The integer whose 1-bits are to be counted.

        Returns:
            int: The number of 1-bits in the binary representation of the input integer.
        """
        return n.bit_count()
    
    ####################################################################################################