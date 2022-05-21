"""
Utilities for Running the app
"""
from typing import List
import more_itertools
import joblib

model = joblib.load('NB_augmented.pkl')
vectorizer = joblib.load('vectorizer.pkl')
categories = [
    'anger',
    'boredom',
     'empty',
     'enthusiasm',
     'fun',
     'happiness',
     'hate',
     'love',
     'neutral',
     'relief',
     'sadness',
     'surprise',
     'worry'
]

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
    vectorized_text = vectorizer.transform([text]).toarray()
    prediction_index = model.predict(vectorized_text)[0]
    print(prediction_index)
    prediction_textual = categories[prediction_index]
    return prediction_textual

if __name__ =='__main__':
    text = "I hate machine learning"
    print(predict(text))