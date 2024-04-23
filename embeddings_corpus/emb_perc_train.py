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



########## FUNCTIONS

def create_test_set(dict_vbs, emb_file):
    neg_emb = []
    pos_emb = []
    neg_lab = []
    pos_lab = []
    for v in dict_vbs:
        for elem in emb_file[v][0]:
            neg_emb.append(elem.numpy())
        for elem in emb_file[v][1]:
            pos_emb.append(elem.numpy())
    print(len(neg_emb))
    print(len(pos_emb))
    lab_set = np.concatenate((np.ones(len(neg_emb)), np.zeros(len(pos_emb))))
    neg_emb = np.array(neg_emb)
    pos_emb = np.array(pos_emb)
    test_set = np.concatenate((neg_emb, pos_emb), 0)
    
    return test_set, lab_set


def train_test_split(data_set, lab_set, random_st_nb):
    data_set, lab_set = skshuffle(data_set, lab_set, random_state = random_st_nb)
    print(f"len dataset pre split: {len(data_set)}")
    train_sz = round(len(data_set)*0.9)
    test_sz = len(data_set)-train_sz
    train_set = data_set[:train_sz]
    train_lab = lab_set[:train_sz]
    test_set = data_set[-test_sz:]
    test_lab = lab_set[-test_sz:]

    return train_set, train_lab, test_set, test_lab










device = torch.device("cuda") if torch.cuda.is_available() else torch.devide("cpu")


print("Loading verb stats...")
verb_stats = pd.read_csv("new_database.csv")
# dataframe 2054 vbs w "lemma", "total occ", "num non neg", "num neg", "perc neg"







########### group of the 100 verbs with lowest neg perc
vb_perc_mn = {}
ls = []
n = 0
for vb in list(verb_stats.head(100)["lemma"]):
    ls.append((vb, n))
    n+=1

for vb, n in ls:
    vb_perc_mn[vb] = list(verb_stats.head(100)["perc neg"])[n]







########### PLOT
#print(list(verb_stats["lemma"][900:1100]))
'''l = list(reversed(range(2054)))
pyplot.scatter(l, list(verb_stats["perc neg"]), marker= ".")
pyplot.ylabel(r"% of negated occurrences")
pyplot.xlabel("Rank")
pyplot.show()'''





########### Extracting embeddings from Milena's file
print("Loading embeddings...")
embs = torch.load(r"/data/vgullace/embeddings990000")
#print(type(embs))
#print(type(embs["offer"][0][0]))
# dictionary
# keys = verb; values = list of lists (list of negative embeddings (tensors) [0], list of positive embeddings [0])



########### group of the 100 verbs with highest neg perc
vb_perc_mx = {}
ls = []
n = 0
for vb in list(verb_stats.tail(100)["lemma"]):
    ls.append((vb, n))
    n+=1

for vb, n in ls:
    vb_perc_mx[vb] = list(verb_stats.tail(100)["perc neg"])[n]


########### Group of verbs with neg perc > 0.1
print("Training on verbs with > 0.1 neg perc")
vbs_oneperc = [v for v in vb_perc_mx.keys() if vb_perc_mx[v]>=0.1]


test_set, lab_set = create_test_set(vbs_oneperc, embs)
train_data, train_labs, test_data, test_labs = train_test_split(test_set, lab_set, 42) 



print("Data normalization...")
########### DATA NORMALISATION

# create scaler
scaler = StandardScaler()
scaler.fit(train_data)
dump(scaler, f"scaler.joblib")


train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

print(f"test data {len(test_data)}")
print(f"test labs {len(test_labs)}")
print(f"train data {len(train_data)}")
print(f"train labs {len(train_labs)}")

'''train_data, train_labs = skshuffle(train_data, train_labs, random_state=42)
test_data, test_labs = skshuffle(test_data, test_labs, random_state=42)
'''






emb_result =[]

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
      print(n)
      clf = MLPClassifier(solver = solv, alpha = a,
                    hidden_layer_sizes=hl, random_state = 1, max_iter=800)

      # train on data
      clf = clf.fit(train_data, train_labs)

      # see predictions on the dataset
      predicted = clf.predict(test_data)
      right_pred = clf.score(test_data, test_labs)
      tn, fp, fn, tp = confusion_matrix(test_labs, predicted).ravel()
      emb_result.append(f"Method\t{solv}\nNb hidden layers\t{str(hl)}\nAlpha\t{str(a)}\nScore\t{right_pred}\nTrue neg\t{tn}\nFalse pos\t{fp}\nFalse neg\t{fn}\nTrue pos\t{tp}\n\n")

     
      dump(clf, f"classifiers/classifier01_{n}.joblib")
    

with open("0.1_TRAIN.txt", "w") as file:
    file.write("EMB 0.1 TRAIN")
with open("0.1_TRAIN.txt", "a") as file:
    for scores in emb_result:
        file.write(scores)

print("EMB > 0.1 TEST\n")
for scores in emb_result:
   print(scores)