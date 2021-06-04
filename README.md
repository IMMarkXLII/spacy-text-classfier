# spacy-text-classfier

1. data-setup.py script reads the input dataset, splits it into training and test datasets, and converts them to the binary doc format recognized by spacy.

2. train the spacy model using the output of the first script and the spacy config file provided alongside using the command `python -m spacy train config.cfg --output ./output2`

3. classifier.py script loads the trained models from the path provided and allows one to test the trained model(outputs a single label score for subjects economics, geography, history, politicalscience, science, sociology).
