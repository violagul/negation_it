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



########## FUNCTIONS ############
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







#device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


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


'''for elem in list(dict_wdnet.keys()):
    classi = list(dict_wdnet.keys())
    classi.remove(elem)
    current = classi
    
    for verb in dict_wdnet[elem]:#ls di verbi
        #print(verb)
        for classe in current:#classe sem
            #for verbs in dict_wdnet[classe]:
                #if verb == verbs:
            if verb in dict_wdnet[classe]:
                new_list2 = list(dict_wdnet[classe]).copy()
                new_list2.remove(verb)
                dict_wdnet[classe] = new_list2
        new_list1 = list(dict_wdnet[elem]).copy()
        new_list1.remove(verb)
        dict_wdnet[elem] = new_list1

print(dict_wdnet)'''





#######################################






'''
verb_repetitions = {}
for verb in verbs_list:
  #print(verb, "\n")
  n=0
  for classe in dict_wdnet.keys():
    if verb in dict_wdnet[classe]:
      #print(classe)
      n+=1
  n = str(n)
  if n in verb_repetitions.keys():
    verb_repetitions[n].append(verb)
  else:
    verb_repetitions[n] = []
    verb_repetitions[n].append(verb)

print(f"classes\t:\tverbs")
for num in sorted(verb_repetitions.keys()):
  print(num, "\t:\t", len(verb_repetitions[num]))


verbs_list = verb_repetitions["1"]
'''







########################################




'''
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
'''




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

with open(f"WDNET_RANKS_RESULTS.txt", "w") as res_file:
    res_file.write(f"WORDNET RANKS RESULTS\n\n")
with open(f"WDNET_RANKS_LIST.txt", "w") as res_file:
    res_file.write(f"WORDNET RANKS LIST\n\n")


for class_set in [["verb.emotion"], ["verb.consumption"], ["verb.cognition"], ["verb.stative"], ["verb.communication"], ["verb.perception"], ["verb.change"], ["verb.creation"], ["verb.competition"], ["verb.weather"], ["verb.contact"], ["verb.motion"]]:
    test_vbs = []
    for classe in class_set:
        for vb in dict_wdnet[classe]:
            if vb not in test_vbs:
                test_vbs.append(vb)



    test_data, test_labs = create_test_set(test_vbs, embs)




    print("Data normalization...")
    ########### DATA NORMALISATION

    scaler = load(f"scaler_wdnet.joblib")


    test_data = scaler.transform(test_data)

    print(f"test data {len(test_data)}, test labs {len(test_labs)}")






    wdnet_result =[]
    accurs = []

    print(f"Testing MLP...")
    # set up the MLP classifier
    # solver : adam or sgd
    # hidden_layer_sizes : 40,40 or 350,350
    # alpha : between 1e-5 and 1e-3
    n=0
    for n in range(1,13):
        clf = load(f"classifiers/classifier_wdnet_{n}.joblib")

        # see predictions on the dataset
        predicted = clf.predict(test_data)
        right_pred = clf.score(test_data, test_labs)
        tn, fp, fn, tp = confusion_matrix(test_labs, predicted).ravel()
        wdnet_result.append(f"Score\t{right_pred}\nTrue neg\t{tn}\nFalse pos\t{fp}\nFalse neg\t{fn}\nTrue pos\t{tp}\n\n")
        accurs.append(right_pred)

        
        
    
    with open(f"WDNET_RANKS_RESULTS.txt", "a") as res_file:
        res_file.write(f"\n{class_set[0][5:]}:\n")
    with open(f"WDNET_RANKS_RESULTS.txt", "a") as res_file:
        for scores in wdnet_result:
            res_file.write(scores)

    
    with open(f"WDNET_RANKS_LIST.txt", "a") as res_file:
        res_file.write(f"\n{class_set[0][5:]}:\n")
    with open(f"WDNET_RANKS_LIST.txt", "a") as res_file:
        for score in accurs[:-1]:
            res_file.write(f"{score}, ")
        res_file.write(f"{accurs[-1]}]\n\n")



    print("WORDNET TEST\n")
    for scores in wdnet_result:
        print(scores)