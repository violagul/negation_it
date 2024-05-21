import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import nltk
#nltk.download("omw-1.4")
import pandas as pd
from nltk.corpus import wordnet
import torch
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle as skshuffle
from joblib import load, dump
from random import shuffle, seed
import statistics as stats



############### FUNCTIONS ##################
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
    lab_set = np.concatenate((np.ones(min(6000,len(neg_emb))), np.zeros(min(6000,len(neg_emb)))))
    shuffle(neg_emb)
    neg_emb = np.array(neg_emb[:6000])
    shuffle(pos_emb)
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

#################################







device = torch.device("cuda") if torch.cuda.is_available() else torch.devide("cpu")


print("Loading verb stats...")
verb_stats = pd.read_csv(f"../embeddings_corpus/new_database.csv")
# dataframe 2054 vbs w "lemma", "total occ", "num non neg", "num neg", "perc neg"




verbs_list = list(verb_stats["lemma"])
#print(verbs_list)
#print(f"\n\nNUMERO VERBI : {len(verbs_list)}\n\n")


dict_wdnet = {}
for vb in verbs_list:
    for syn in wordnet.synsets(vb):
        if "verb." in syn.lexname():
            classe = syn.lexname()
            if classe in dict_wdnet:
                dict_wdnet[classe].append(vb)
            else:
                dict_wdnet[classe] = []
                dict_wdnet[classe].append(vb)



for elem in dict_wdnet.keys():
    verb_ls = list(set(dict_wdnet[elem]))
    dict_wdnet[elem] = verb_ls
    #print(f"{elem}\t{len(dict_wdnet[elem])}\n")



verb_stats = verb_stats.sort_values(by = "perc neg", ascending = False, ignore_index = True)
#vb_lemmas = verb_stats.set_index("lemma")

mean_perc_neg = {}

for classe in dict_wdnet.keys():
    percentages = []
    for vb in dict_wdnet[classe]:
        row = (verb_stats["lemma"] == vb)
        perc_neg = float(verb_stats.loc[row, "perc neg"])
        percentages.append(perc_neg)
    mean_perc_neg[classe] = stats.mean(percentages)

'''for key in mean_perc_neg.keys():
    print(f"{key}\n{mean_perc_neg[key]}\n\n")'''


mean_perc = pd.DataFrame({
    "VerbClass": mean_perc_neg.keys(),
    "MeanNegPerc": mean_perc_neg.values()
}).sort_values(by = "MeanNegPerc", ascending = False, ignore_index = True)






########### Extracting embeddings from Milena's file
print("Loading embeddings...")
embs = torch.load(r"/data/vgullace/embeddings990000")
#print(type(embs))
#print(type(embs["offer"][0][0]))
# dictionary
# keys = verb; values = list of lists (list of negative embeddings (tensors) [0], list of positive embeddings [0])





#### training on verb.social, verb.possession, verb.body

train_vbs = []
for classe in ["verb.social", "verb.possession", "verb.body"]:
    for vb in dict_wdnet[classe]:
        if vb not in train_vbs:
            train_vbs.append(vb)



test_set, lab_set = create_test_set(train_vbs, embs)
train_data, train_labs, test_data, test_labs = train_test_split(test_set, lab_set, 42)




print("Data normalization...")
########### DATA NORMALISATION

# create scaler
scaler = StandardScaler()
scaler.fit(train_data)
dump(scaler, f"scaler_wdnet.joblib")


train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

print(f"test data {len(test_data)}, test labs {len(test_labs)}\ntrain data {len(train_data)}, train labs {len(train_labs)}")



wdnet_result =[]
accurs = []

print(f"Training and testing MLP...")
# set up the MLP classifier
# solver : adam or sgd
# hidden_layer_sizes : 40,40 or 350,350
# alpha : between 1e-5 and 1e-3
n=0
'''for hl in [(350,350),(40,40)]:
  for a in [1e-5, 1e-4, 1e-3]:
    for solv in ["adam", "sgd"]:
      n+=1
      print(n)
      clf = MLPClassifier(solver = solv, alpha = a,
                    hidden_layer_sizes=hl, random_state = 1, max_iter=400)

      # train on data
      clf = clf.fit(train_data, train_labs)

      # see predictions on the dataset
      predicted = clf.predict(test_data)
      right_pred = clf.score(test_data, test_labs)
      tn, fp, fn, tp = confusion_matrix(test_labs, predicted).ravel()
      wdnet_result.append(f"Method\t{solv}\nNb hidden layers\t{str(hl)}\nAlpha\t{str(a)}\nScore\t{right_pred}\nTrue neg\t{tn}\nFalse pos\t{fp}\nFalse neg\t{fn}\nTrue pos\t{tp}\n\n")
      accurs.append(right_pred)

     
      dump(clf, f"classifiers/classifier_wdnet_{n}.joblib")'''

for rand_st in [1, 42, 700]:
    clf = MLPClassifier(solver = adam, alpha = 0.001, hidden_layer_sizes=(40, 40), random_state = rand_st, max_iter=500)
    clf = clf.fit(train_data, train_labs)

    # see predictions on the dataset
    predicted = clf.predict(test_data)
    right_pred = clf.score(test_data, test_labs)
    tn, fp, fn, tp = confusion_matrix(test_labs, predicted).ravel()
    emb_result.append(f"Method\t{solv}\nNb hidden layers\t{str(hl)}\nAlpha\t{str(a)}\nScore\t{right_pred}\nTrue neg\t{tn}\nFalse pos\t{fp}\nFalse neg\t{fn}\nTrue pos\t{tp}\n\n")

     
    dump(clf, f"classifiers/classifier_wdnet_rs{rand_st}.joblib")




with open(f"WDNET_TRAIN.txt", "w") as res_file:
    res_file.write("WORDNET TRAIN\n")
with open(f"WDNET_TRAIN.txt", "a") as res_file:
    for scores in wdnet_result:
        res_file.write(scores)

with open(f"WDNET_TRAIN_LIST.txt", "w") as res_file:
    res_file.write("WORDNET TRAIN\n[")
with open(f"WDNET_TRAIN_LIST.txt", "a") as res_file:
    for score in accurs[:-1]:
        res_file.write(f"{score}, ")
    res_file.write(f"{accurs[-1]}]\n\n")



print("WORDNET TEST\n")
for scores in wdnet_result:
   print(scores)