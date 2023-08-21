
#
# Date: 21 August 2023
#
# Author: Andrew Rigby
#
# Purpose:
#

import numpy as np
import matplotlib.pyplot as plt


def round_up_to_odd(f):
    """
    Purpose:
        Round a floating point number up to the nearest odd integer
    """
    return np.ceil(f) // 2 * 2 + 1