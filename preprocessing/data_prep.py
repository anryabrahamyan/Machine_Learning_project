"""
File for preprocessing and augmenting data
"""
import pandas as pd

dataset = pd.read_csv("../datasets/tweet_emotions.csv").drop(['tweet_id'], inplace=True, axis=1)

# TODO preprocessing for tweets
