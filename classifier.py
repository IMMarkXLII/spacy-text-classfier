import os

import spacy

PATH = os.path.abspath(os.curdir)
MODEL_PATH = f"{PATH}/saved_models/spacy/model-best"
NLP = spacy.load(MODEL_PATH)


def classify(text: str) -> dict:
    """
    predicts the subject for the given text based on the best model of spacy
    :param text: the text to classify
    :return: the classification of the provided text
    """
    doc = NLP(text)
    sorted_results = dict(sorted(doc.cats.items(), key=lambda x: x[1], reverse=True))
    return sorted_results
