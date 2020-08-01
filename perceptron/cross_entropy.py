import numpy as np


# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(classification, probabilities):
    classification = np.float_(classification)
    probabilities = np.float_(probabilities)
    return -np.sum(
        classification * np.log(probabilities)
        + (1 - classification) * np.log(1 - probabilities)
    )


if __name__ == "__main__":
    y = [1, 0, 1, 1]
    p = [0.4, 0.6, 0.1, 0.5]
    expected = 4.828313737302301
    result = cross_entropy(classification=y, probabilities=p)
    print(result)
    assert expected == result
