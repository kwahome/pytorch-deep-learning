import numpy as np


# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(data):
    exponents = np.exp(data)
    total = exponents.sum()
    return [float(item) / total for item in exponents]

    # Note: The function np.divide can also be used here, as follows:
    # expL = np.exp(L)
    # return np.divide (expL, expL.sum())


if __name__ == "__main__":
    test_data = [5, 6, 7]
    expected = [0.09003057317038046, 0.24472847105479764, 0.6652409557748219]
    result = softmax(test_data)
    print(result)
    assert(expected == result)
