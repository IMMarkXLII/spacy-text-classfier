import os
from copy import deepcopy

import spacy
from sklearn import model_selection
from spacy.tokens import DocBin

# DocBin is spacys new way to store Docs in a
# binary format for training later

subjects = {
    "economics": 0,
    "geography": 0,
    "history": 0,
    "politicalscience": 0,
    "science": 0,
    "sociology": 0
}

nlp = spacy.load('en_core_web_md')

targetDirPath = "~/Downloads/targetDataset"

all_files = [f'{targetDirPath}/{f}' for f in os.listdir(targetDirPath) if '.txt' in f]

lines = []
for file_path in all_files:
    with open(file_path, 'r') as f:
        all_lines = f.readlines()
        subject = file_path.split("/")[5].replace(".txt", "")
        all_line_tuples = [(line, subject) for line in all_lines]
        lines.extend(all_line_tuples)

docs = []
for doc, subject in nlp.pipe(lines, as_tuples=True):
    labels = deepcopy(subjects)
    labels[subject] = 1
    doc.cats = labels
    docs.append(doc)

test_dataset, training_dataset = model_selection.train_test_split(docs,
                                                                  train_size=0.5,
                                                                  test_size=0.5)

doc_bin = DocBin(docs=test_dataset)
doc_bin.to_disk("./data/train1.spacy")

doc_bin = DocBin(docs=test_dataset)
doc_bin.to_disk("./data/valid1.spacy")
