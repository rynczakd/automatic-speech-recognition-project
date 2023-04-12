# utils.py
import math
import numpy as np


# Convert Hz to Mel scale
def hz2mel(hz):
    return 2595 * np.log10(1 + hz / 700.0)


# Convert Mel scale to Hz
def mel2hz(mel):
    return 700 * (10 ** (mel / 2595.0) - 1)


# Calculate first power of 2
def first_power_of_2(n: int):
    a = int(math.log2(n))

    if np.power(2, a) == n:
        return int(n)

    return int(np.power(2, (a + 1)))

