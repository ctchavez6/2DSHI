import sys
from enum import Enum

class CONSTANTS(Enum):
    SIXTEEN_BIT_MAX = (2 ** 16) - 1
    TWELVE_BIT_MAX = (2 ** 12) - 1
    EIGHT_BIT_MAX = (2 ** 8) - 1
    EPSILON = sys.float_info.epsilon  # Smallest possible difference

class STEP_DESCRIPTIONS(Enum):
    S01_DESC = "Step 1 - Stream Raw Camera Feed"
    S02_DESC_PREV_WARP_MATRIX = "Step 2 - You created a Warp Matrix 1 last run. Would you like to use it?"
    S02_DESC_NO_PREV_WARP_MATRIX = "Step 2 - New Co-Registration with with Euclidean Transform"
    S03_DESC = "Step 3 - Set Gaussian-Based Static Centers"
    S04_DESC = "Step 4 - Define Regions of Interest"
    S05_DESC = "Step 5 - Close in on ROI?"
    S06_DESC = "Step 6 - Commence Image Algebra (Free Stream):"
    S07_DESC = "Step 7 - Image Algebra (Record): Proceed"
    S08_DESC = "Step 8 - Write Recorded R Frame(s) to File(s)?"
    S09_DESC = "Step 9 - Write some notes to a file?"
