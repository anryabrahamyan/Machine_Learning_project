"""
File for training the model
"""
#TODO fill in the rest
import argparse
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import pandas as pd

# Parse input arguments for the model training
parser = argparse.ArgumentParser(description='Arguments for model training.')
parser.add_argument('-data_path', type=str, default='../datasets/preprocessed_data.csv',
                    help='path to where the data is stored.')


def load_data(path: str) -> pd.DataFrame:
    """
    load the data
    :param path: path to the data
    :return: DataFrame: DataFrame with the data in it
    """

    return pd.read_csv(path)

def create_model(data, max_length):
    """Function for creating the transformer model"""
    pass


def train_model(*args):
    """Function for training the model and saving it in the models file."""
    pass

def load_model(*args):
    """test function for loading the model after saving"""
# Load RoBERTa's tokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-base")  # Tokenizer
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='tf')  # Tokenized text

# Load and compile the model
model = TFAutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=n_categories)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5, clipnorm=1.),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.metrics.SparseCategoricalAccuracy(),
             tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='Sparse_Top_3_Categorical_Accuracy')],
)
#TODO convert model to class
if __name__ == '__main__':
    args = parser.parse_args()
    data = load_data(args.data_path)
