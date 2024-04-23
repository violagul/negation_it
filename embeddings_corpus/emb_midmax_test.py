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




########### group of the 100 verbs with highest neg perc
'''vb_perc_mx = {}
ls = []
n = 0
for vb in list(verb_stats.tail(100)["lemma"]):
    ls.append((vb, n))
    n+=1

for vb, n in ls:
    vb_perc_mx[vb] = list(verb_stats.tail(100)["perc neg"])[n]'''




vbs_ranks = {}
ls = {}
#ranges = [(1,100), (101, 200), (201, 300), (301,400), (401, 500), (501, 600), (601, 700), (701,800), (1201, 1300), (1301, 1400), (1401, 1500), (1501, 1600), (1601, 1700), (1701, 1800), (1801, 1900), (1901, 2000), (2001, 2054)]
#ranges = [(1,50), (51,100), (101, 150), (151, 200), (201, 250),(251, 300), (301, 350), (351, 400), (401, 450), (451,500), (501,550) ,(551,600), (601, 650), (651,700), (701,750), (751,800), (1201, 1250), (1251,1300), (1301, 1350), (1351,1400), (1401, 1450), (1451,1500), (1501, 1550), (1551,1600), (1601, 1650), (1651,1700), (1701, 1750), (1751, 1800), (1801, 1850),(1851, 1900), (1901, 1950), (1951, 2000), (2001, 2054)]
#ranges = [(1,50), (51,100), (101, 150), (151, 200), (201, 250),(251, 300), (301, 350), (351, 400), (401, 450), (451,500), (501,550) ,(551,600), (601, 650), (651,700), (701,750), (751,800), (1201, 1250), (1251,1300), (1301, 1350), (1351,1400), (1401, 1450), (1451,1500), (1501, 1550), (1551,1600), (1601, 1650), (1651,1700), (1701, 1750), (1751, 1800), (1801, 1850),(1851, 1900), (1901, 1950), (1951, 2000), (2001, 2050)]
ranges = [(1, 25), (26, 50), (51, 75), (76,100), (101,125),(126, 150), (151,175),  (176,200), (201, 225), (226, 250),(251, 275), (276, 300), (301, 325), (326, 350), (351, 375), (376, 400), (401, 425), (426, 450), (451,475), (476, 500), (501, 525), (526, 550) ,(551,575), (576, 600), (601, 625), (626, 650), (651, 675), (676, 700), (701, 725), (726, 750), (751, 775), (776, 800), (1201, 1225), (1226, 1250), (1251, 1275), (1276, 1300), (1301, 1325), (1326, 1350), (1351, 1375), (1376, 1400), (1401, 1425), (1426, 1450), (1451, 1475), (1476, 1500), (1501, 1525), (1526, 1550), (1551, 1575), (1576, 1600), (1601, 1625), (1626, 1650), (1651, 1675), (1676, 1700), (1701, 1725), (1726, 1750), (1751, 1775), (1776, 1800), (1801, 1825), (1826, 1850),(1851, 1875), (1876, 1900), (1901, 1925), (1926, 1950), (1951, 1975), (1976, 2000), (2001, 2025), (2026, 2050)]
#for vb in list(verb_stats["lemma"])[1001:1100]:
for m, l in ranges:
    n = 0
    #for vb in list(verb_stats["lemma"])[801:1200]:
    partial = []
    for vb in list(verb_stats["lemma"])[m:l]:
        partial.append((vb, n))
        n+=1
    ls[l] = partial
    #print(partial)


#print(ls.items())
for rank, sets in ls.items():
    #print(sets)
    #print(rank)
    verbs = {}
    for vb, n in sets:
        #print(vb)
        #verbs[vb] = list(verb_stats["perc neg"])[l-99:l][n]
        verbs[vb] = list(verb_stats["perc neg"])[l-24:l][n]
        
    #verbs[sets[0]] = list(verb_stats["perc neg"])[l-99:l][sets[1]]
    vbs_ranks[rank] = verbs


'''
print(str(vbs_ranks[300]))
print(str(vbs_ranks[600]))'''




########### Extracting embeddings from Milena's file
print("Loading embeddings...")
embs = torch.load(r"/data/vgullace/embeddings990000")
#print(type(embs))
#print(type(embs["offer"][0][0]))
# dictionary
# keys = verb; values = list of lists (list of negative embeddings (tensors) [0], list of positive embeddings [0])





scaler = load(f"scaler_cent.joblib")

'''with open(f"MID_50ALLRANKS_TEST.txt", "w") as res_file:
    res_file.write("MID 50 ALL-RANKS TEST\n")

with open(f"MID_50ALLRANKS_LISTS.txt", "w") as acc_file:
    acc_file.write("LIST")
'''


with open(f"MID_25ALLRANKS_TEST.txt", "w") as res_file:
    res_file.write("MID 25 ALL-RANKS TEST\n")

with open(f"MID_25ALLRANKS_LISTS.txt", "w") as acc_file:
    acc_file.write("LIST")


for rank in vbs_ranks.keys():
    vb_dict = vbs_ranks[rank]
    test_data, test_labs = create_test_set(vb_dict, embs)
    #train_data, train_labs, test_data, test_labs = train_test_split(test_set, lab_set, 42) 



    print("Data normalization...")
    ########### DATA NORMALISATION

    


    test_data = scaler.transform(test_data)

    print(f"test data {len(test_data)}")
    print(f"test labs {len(test_labs)}")


    emb_result =[]
    accurs = []




    print(f"Testing MLP rank {rank}...")
    # set up the MLP classifier
    # solver : adam or sgd
    # hidden_layer_sizes : 40,40 or 350,350
    # alpha : between 1e-5 and 1e-3
    n=0
    for n in range(1,13):
        clf = load(f"classifiers/classifier_cent_{n}.joblib")
        
        # see predictions on the dataset
        predicted = clf.predict(test_data)
        right_pred = clf.score(test_data, test_labs)
        tn, fp, fn, tp = confusion_matrix(test_labs, predicted).ravel()
        emb_result.append(f"Score\t{right_pred}\nTrue neg\t{tn}\nFalse pos\t{fp}\nFalse neg\t{fn}\nTrue pos\t{tp}\n\n")
        accurs.append(f"{right_pred}")

        
    with open(f"MID_25ALLRANKS_TEST.txt", "a") as res_file:
        res_file.write(f"{rank-24}-{rank} ranks\n")
    with open(f"MID_25ALLRANKS_TEST.txt", "a") as res_file:
        for scores in emb_result:
            res_file.write(scores)

    with open(f"MID_25ALLRANKS_LISTS.txt", "a") as acc_file:
        acc_file.write(f"{rank-24}-{rank} ranks\n[")
        for score in accurs[:-1]:
            acc_file.write(f"{score}, ")
        acc_file.write(f"{accurs[-1]}]\n\n")


    print(f"MID TEST 25 RANK {rank}\n")
    for scores in emb_result:
        print(scores)
    
    
    
    
'''




test_data, test_labs = create_test_set(vb_perc_mx, embs)
#train_data, train_labs, test_data, test_labs = train_test_split(test_set, lab_set, 42) 



print("Data normalization...")
########### DATA NORMALISATION

scaler = load(f"scaler_cent.joblib")


test_data = scaler.transform(test_data)

print(f"test data {len(test_data)}")
print(f"test labs {len(test_labs)}")




emb_result =[]

print(f"Training and testing MLP...")
# set up the MLP classifier
# solver : adam or sgd
# hidden_layer_sizes : 40,40 or 350,350
# alpha : between 1e-5 and 1e-3
n=0
for n in range(1,13):
    clf = load(f"classifiers/classifier_cent_{n}.joblib")
    
    # see predictions on the dataset
    predicted = clf.predict(test_data)
    right_pred = clf.score(test_data, test_labs)
    tn, fp, fn, tp = confusion_matrix(test_labs, predicted).ravel()
    emb_result.append(f"Score\t{right_pred}\nTrue neg\t{tn}\nFalse pos\t{fp}\nFalse neg\t{fn}\nTrue pos\t{tp}\n\n")


    
with open(f"MIDMAX_TEST.txt", "w") as res_file:
    res_file.write("MIDMAX TEST\n")
with open(f"MIDMAX_TEST.txt", "a") as res_file:
    for scores in emb_result:
        res_file.write(scores)



print("MIDMAX TEST\n")
for scores in emb_result:
   print(scores)

'''