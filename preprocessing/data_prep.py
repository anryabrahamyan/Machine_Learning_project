"""
File for preprocessing and augmenting data
"""
import more_itertools
import pandas as pd
import argparse


# TODO add argparse
# TODO implement sliding window
# TODO add tweet preprocessing here
# TODO add NLP augmentation
# TODO evaluate model and choose which features to keep
# TODO ADD requirements at the end

def faster_window(string: str, window: int = 514, step: int = 1):
    """
    A function for split a sentence of length>window to parts
    :param string: str An input sequence to be checked and split if necessary
    :param window: int The size of the window
    :param step: int An optional step parameter
    :return: A list of strings
    """
    if len(string.split(" ")) > window:
        input = string.split(" ")
        return [" ".join(i) for i in list(more_itertools.windowed(input,
                                                                  n=window,
                                                                  step=step))]
    else:
        return [string]


dataset = pd.read_csv("../datasets/preprocessed_tweets.csv").drop(['tweet_id'],
                                                                  inplace=True, axis=1)

# TODO preprocessing for tweets
