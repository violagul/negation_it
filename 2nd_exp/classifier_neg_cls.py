# -*- coding: utf-8 -*-
"""classifier_neg_cls.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1NiZjC1IHQGuH3GvkV88CE0jur9GnUrZj
"""

! pip install datasets transformers

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import re

model = AutoModel.from_pretrained('dbmdz/bert-base-italian-cased') # automodel for masked LM perché automodel e basta crea solo i vettori, gli embedding, per la frase; per LM invece ricava anche le prob di ogni parola nel vocab, ossia fa il language model
tokenizer = AutoTokenizer.from_pretrained('dbmdz/bert-base-italian-cased')

with open(r"paisa.raw.utf8", encoding='utf8') as infile:
    paisa = infile.read()

wiki_pattern = r"<text.*wiki.*(?:\n.*)+?\n</text>\n" # finds all texts containing "wiki" in their tag's url
paisa_wiki = re.findall(wiki_pattern, paisa)
#print(f"Number of texts from a site containing 'wiki' in their URL: {len(paisa_wiki)}")

sent = []
pattern = r" [A-Z][a-z ]*[,:]?[a-z ]+[,:]?[a-z ][,:]?[a-z]+\. \b"  # finds kind of acceptable sentences

for text in paisa_wiki:
  found = re.findall(pattern, text)
  for elem in  found:
    if len(elem) > 25:
      sent.append(elem)

#print(f"Number of sentences: {len(sent)}")

sent = []
pattern = r" [A-Z][a-z ]*[,:]?[a-z ]+[,:]?[a-z ][,:]?[a-z]+\. \b"  # finds kind of acceptable sentences

for text in paisa_wiki:
  found = re.findall(pattern, text)
  for elem in  found:
    if len(elem) > 25:
      sent.append(elem)

#print(f"Number of sentences: {len(sent)}")

for sent_list in [sent_neg, sent_pos]:
  batch_encoded = tokenizer.batch_encode_plus(sent_list, padding=True, add_special_tokens=True, return_tensors="pt")

  with torch.no_grad():
    tokens_outputs = model(**batch_encoded)

  cls_encodings = tokens_outputs.last_hidden_states[:,0,:]
  cls_encodings = cls_encodings_pos.cpu().numpy()

  if sent_list == sent_neg:
    cls_encodings_neg = cls_encodings
  elif sent_list == sent_pos:
    cls_encodings_pos = cls_encodings

train = cls_encodings_neg[:9000]
train = train.append(cls_encodings_pos[:9000])
test = cls_encodings_pos[9000:]
test = test.append(cls_encodings_neg[9000:])
labels = np.empty(18000)
labels = np.where(labels[:9000], 1, 0)

scaler = StandardScaler()
scaler.fit(dati.values)

dati_scaled = scaler.transform(dati.values)
dati_scaled

train = dati_scaled
labels = df["class"].values

X = train
y = labels

# solver : adam o sgd
# hidden_layer_sizes : testare diverse
# alpha : tra 1e-5 e 1e-2
clf = MLPClassifier(solver = "adam", alpha = 1e-3,
                    hidden_layer_sizes=(40, 2), random_state = 1)

clf.fit(X, y)

clf.predict(test)