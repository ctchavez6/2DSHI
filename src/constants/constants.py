import sys
from enum import Enum

class CONSTANTS(Enum):
    SIXTEEN_BIT_MAX = (2 ** 16) - 1
    TWELVE_BIT_MAX = (2 ** 12) - 1
    EIGHT_BIT_MAX = (2 ** 8) - 1
    EPSILON = sys.float_info.epsilon  # Smallest possible difference
