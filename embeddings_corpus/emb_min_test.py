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
    lab_set = np.concatenate((np.ones(len(neg_emb)), np.zeros(len(neg_emb))))
    neg_emb = np.array(neg_emb)
    pos_emb = np.array(pos_emb[:len(neg_emb)])
    print(f"neg emb {len(neg_emb)}")
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













vbs_ranks = {}
ls = {}
ranges = [(1,100), (101, 200), (201, 300), (301,400), (401, 500), (501, 600), (601, 700), (701,800), (801,900), (901, 1000), (1001, 1100), (1101, 1200), (1201, 1300), (1301, 1400), (1401, 1500), (1501, 1600), (1601, 1700), (1701, 1800), (1801, 1900), (1901, 2000), (2001, 2054)]
#for vb in list(verb_stats["lemma"])[1001:1100]:
for m, l in ranges:
    n = 0
    #for vb in list(verb_stats["lemma"])[801:1200]:
    partial = []
    for vb in list(verb_stats["lemma"])[m:l]:
        partial.append((vb, n))
        n+=1
    ls[l] = partial


#print(ls.items())
for rank, sets in ls.items():
    #print(sets)
    #print(rank)
    verbs = {}
    for vb, n in sets:
        #print(vb)
        verbs[vb] = list(verb_stats["perc neg"])[l-99:l][n]
    #verbs[sets[0]] = list(verb_stats["perc neg"])[l-99:l][sets[1]]
    vbs_ranks[rank] = verbs






with open(f"MIN_ALLRANKS_TEST.txt", "w") as file:
    file.write("MIN ALL-RANKS TEST\n\n")

with open(f"MIN_ALLRANKS_LIST.txt", "w") as file:
    file.write("MIN ALL-RANKS LISTS\n\n")


scaler = load(f"scaler_min.joblib")

########### Extracting embeddings from Milena's file
print("Loading embeddings...")
embs = torch.load(r"/data/vgullace/embeddings990000")
#print(type(embs))
#print(type(embs["offer"][0][0]))
# dictionary
# keys = verb; values = list of lists (list of negative embeddings (tensors) [0], list of positive embeddings [0])

for rank in vbs_ranks.keys():
    vb_dict = vbs_ranks[rank]
    test_data, test_labs = create_test_set(vb_dict, embs)

    test_data, test_labs = create_test_set(vb_dict, embs)
    #train_data, train_labs, test_data, test_labs = train_test_split(test_set, lab_set, 42) 



    print("Data normalization...")
    ########### DATA NORMALISATION

    

    #dati_scaled = scaler.transform(train_data)
    test_data = scaler.transform(test_data)

    print(f"test data {len(test_data)}")
    print(f"test labs {len(test_labs)}")
    '''print(f"train data {len(train_data)}")
    print(f"train labs {len(train_labs)}")'''




    emb_result =[]
    accurs = []

    print(f"Training and testing MLP...")
    # set up the MLP classifier
    # solver : adam or sgd
    # hidden_layer_sizes : 40,40 or 350,350
    # alpha : between 1e-5 and 1e-3
    n=0
    for n in range(1,13):
        clf = load(f"classifiers/classifier_min_{n}.joblib")

        # see predictions on the dataset
        predicted = clf.predict(test_data)
        right_pred = clf.score(test_data, test_labs)
        tn, fp, fn, tp = confusion_matrix(test_labs, predicted).ravel()
        emb_result.append(f"Score\t{right_pred}\nTrue neg\t{tn}\nFalse pos\t{fp}\nFalse neg\t{fn}\nTrue pos\t{tp}\n\n")
        accurs.append(f"{right_pred}")

        

    with open(f"MIN_ALLRANKS_TEST.txt", "w") as res_file:
        res_file.write(f"{rank-99}--{rank} ranks\n")
    with open(f"MIN_ALLRANKS_TEST.txt", "a") as res_file:
        for scores in emb_result:
            res_file.write(scores)
    
    with open(f"MIN_ALLRANKS_LIST.txt", "a") as acc_file:
        acc_file.write(f"{rank-99}-{rank} ranks\n[")
        for score in accurs[:-1]:
            acc_file.write(f"{score}, ")
        acc_file.write(f"{accurs[-1]}]\n\n")



    print("MIN TEST\n")
    for scores in emb_result:
        print(scores)