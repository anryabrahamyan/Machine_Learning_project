import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd


def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])


if __name__ == "__main__":
    dataset = pd.read_csv("../datasets/train_preprocessed.csv")
    X,y = dataset["content"],dataset["sentiment"]
    X = tf.convert_to_tensor(X,dtype=tf.str)
    print(X)