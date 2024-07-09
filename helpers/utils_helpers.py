import numpy as np
from contextlib import contextmanager
import sys
import os


def calculate_percentage_within_ci(test_data, conf_int):
    """
    Calculate the proportion of actual values that fall within the confidence intervals.

    Parameters:
    -----------
    test_data : array-like
        Array of actual values.
    conf_int : array-like
        Array of confidence intervals.

    Returns:
    --------
    float
        Proportion of actual values within the confidence intervals.
    """
    lower_bounds = conf_int[:, 0]
    upper_bounds = conf_int[:, 1]
    within_ci = np.sum((test_data >= lower_bounds) & (test_data <= upper_bounds))
    proportion_in_ci = within_ci / len(test_data)
    return proportion_in_ci

#To suppress the verbose output from AutoTS during model training, you can redirect the standard output and standard error streams temporarily 
#while fitting the model. This approach will prevent the print statements from being displayed on the console.
@contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def check_constant_prediction(predictions):
    """
    Function to check if predicted values are a flat line (constant).
    
    Parameters:
    predictions : array-like
        Predicted values from the model.
    
    Returns:
    bool
        True if predictions are a flat line, False otherwise.
    """
    # Convert predictions to numpy array if not already
    predictions = np.asarray(predictions)
    
    # Check if all predicted values are exactly the same
    if np.all(predictions == predictions[0]):
        return True
    else:
        return False