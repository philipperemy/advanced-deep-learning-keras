import numpy as np


def cost_function_regression(targets, predictions):
    return np.mean(np.square(np.subtract(targets, predictions)))


def cost_function_classification(targets, predictions):
    return np.mean(-targets * np.log(predictions))


if __name__ == '__main__':
    targets = np.array([1, 2, 3])
    predictions = np.array([0, 1, 8])

    print('targets             =', targets)
    print('predictions         =', predictions)

    print('Regression cost = ', cost_function_regression(targets, predictions))

    CLASSES = {
        0: 'cat',
        1: 'dog'
    }

    targets = np.array([0, 1, 1])  # cat, dog, dog
    good_predictions = np.array([0.1, 0.9, 0.9])  # predicted probability distribution
    bad_predictions = np.array([0.9, 0.1, 0.2])  # predicted probability distribution

    print('targets              =', targets)
    print('good predictions     =', good_predictions)
    print('bad predictions      =', good_predictions)

    print('Classification cost (good) = ', cost_function_classification(targets, good_predictions))
    print('Classification cost (bad)  = ', cost_function_classification(targets, bad_predictions))
