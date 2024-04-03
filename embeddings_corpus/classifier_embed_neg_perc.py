import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


import torch
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle as skshuffle
from joblib import load, dump
from random import shuffle, seed
import pandas as pd

import matplotlib.pyplot as pyplot



#device = torch.device("cuda") if torch.cuda.is_available() else torch.devide("cpu")


embs = torch.load(r"/data/vgullace/embeddings990000")
# dictionary
# keys = verb; velues = list of lists (list of negative embeddings [0], list of positive embeddings [0])

verb_stats = pd.read_csv("new_database.csv")
# dataframe 2054 vbs w "lemma", "total occ", "num non neg", "num neg", "perc neg"

vb_perc_mx = {}
ls = []
n = 0
for vb in list(verb_stats.tail(100)["lemma"]):
    ls.append((vb, n))
    n+=1

for vb, n in ls:
    vb_perc_mx[vb] = list(verb_stats.tail(100)["perc neg"])[n]

vb_perc_mn = {}
ls = []
n = 0
for vb in list(verb_stats.head(100)["lemma"]):
    ls.append((vb, n))
    n+=1

for vb, n in ls:
    vb_perc_mn[vb] = list(verb_stats.head(100)["perc neg"])[n]



vb_perc_cent = {}
ls = []
n = 0
for vb in list(verb_stats["lemma"])[901:1100]:
    ls.append((vb, n))
    n+=1

for vb, n in ls:
    vb_perc_cent[vb] = list(verb_stats["perc neg"])[901:1100][n]





#print(list(verb_stats["lemma"][900:1100]))
'''l = list(reversed(range(2054)))
pyplot.scatter(l, list(verb_stats["perc neg"]), marker= ".")
pyplot.ylabel(r"% of negated occurrences")
pyplot.xlabel("Rank")
pyplot.show()'''



# prendere un gruppo di verbi
# prendere i corrispondenti mbeddings divisi per neg e pos
# scaler
# train classifier
# prendere altri gruppi, scaler e testare


vbs_oneperc = [v for v in vb_perc_mx.keys() if vb_perc_mx[v]>=0.1]

neg_embs = []
neg_lab = []
pos_embs = []
pos_lab = []
for v in vbs_oneperc:
   for elem in embs[v][1]:
      pos_embs.append(elem)
      pos_embs.append(0)
   for elem in embs[v][0]:
      neg_embs.append(elem)
      neg_embs.append(1)

print(neg_lab)
print(neg_embs)






'''
paisa_result =[]

print(f"Training and testing MLP...")
# set up the MLP classifier
# solver : adam or sgd
# hidden_layer_sizes : 40,40 or 350,350
# alpha : between 1e-5 and 1e-3
n=0
for hl in [(350,350),(40,40)]:
  for a in [1e-5, 1e-4, 1e-3]:
    for solv in ["adam", "sgd"]:
      n+=1
      clf = MLPClassifier(solver = solv, alpha = a,
                    hidden_layer_sizes=hl, random_state = 1, max_iter=500)

      # train on data
      clf = clf.fit(X, y)

      # see predictions on the dataset
      predicted = clf.predict(test)
      right_pred = clf.score(test, test_lab)
      tn, fp, fn, tp = confusion_matrix(test_lab, predicted).ravel()
      paisa_result.append(f"Method\t{solv}\nNb hidden layers\t{str(hl)}\nAlpha\t{str(a)}\nScore\t{right_pred}\nTrue neg\t{tn}\nFalse pos\t{fp}\nFalse neg\t{fn}\nTrue pos\t{tp}\n\n")

     
      dump(clf, f"../Inputs/non_classifier_{n}.joblib")
    

print("PAISA' TEST\n")
for scores in paisa_result:
   print(scores)'''