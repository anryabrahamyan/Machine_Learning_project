"""
Utilities for Running the app
"""
from typing import List
import more_itertools

# sliding window for strings that are too long
def sliding_window(string: str, window: int = 514, step: int = 1) -> List[str]:
    """
    A function for split a sentence of length>window to parts
    :param string: str An input sequence to be checked and split if necessary
    :param window: int The size of the window
    :param step: int An optional step parameter
    :return: A list of strings
    """
    if len(string.split(" ")) > window:
        text_input = string.split(" ")
        return [" ".join(i) for i in list(more_itertools.windowed(text_input,
                                                                  n=window,
                                                                  step=step))]
    else:
        return [string]



def predict(text):
    #TODO change later to model
    return text