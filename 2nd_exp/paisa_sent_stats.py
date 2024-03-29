import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch
from joblib import load, dump
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle as skshuffle
import re
from random import shuffle, seed
import matplotlib.pyplot as pyplot
import matplotlib
import statistics as stat
from math import log



#from sklearn.model_selection import train_test_split

from tools.build_array import build_array, build_hypo, build_masked_context
from tools.chech_conjug import check_conjugation, get_conj

#from tools.chech_conjug import check_conjugation

device = torch.device("cuda") if torch.cuda.is_available() else torch.devide("cpu")






########################
### useful functions ###
########################







def make_and_encode_batch(current_batch, tokenizer, model, device, batch_verbs, name_available, profession_available, current_pronouns_maj, found):
    current_found = found # true if the current batch contained good sentences
    good_pred = 0
    detail_verbs = []

    # get the predicted tokens for the batch of sentences
    predictions = encode_batch(current_batch, tokenizer, model, device)
    new_sentence = None

    # for each prediction, check if the model predicted the same verb that was in the context sentence 
    for i, prediction_available in enumerate(predictions):
        good_verb = batch_verbs[i] # the desired verb

        if check_conjugation(good_verb, prediction_available):
            # outputs True value if the prediction is the 3rd person plural of the desired verb
            detail_verbs.append(good_verb)
            good_pred += 1
            good_dico = {"name_available": name_available, "profession_available": profession_available,
                         "verb": good_verb, "current_pronouns_maj": current_pronouns_maj, "masked_prediction": prediction_available}

            if not current_found:
                # once a good sentence is found, the "found" value is set to true
                # and the "new_sentence" value is the dictionary of all elements in the sentence
                new_sentence = good_dico

                current_found = True
                #if not complete_check: ########
                #    break
    return new_sentence, current_found, good_pred, detail_verbs




def encode_batch(current_batch, tokenizer, model, device):

    with torch.no_grad():
        # encode sentences
        encoded_sentence = tokenizer.batch_encode_plus(current_batch,padding=True,  return_tensors="pt").to(device)
        # get the mask-token index in the sentence 
        mask_tokens_index = torch.where(encoded_sentence['input_ids'] == tokenizer.mask_token_id) 
        #print(mask_tokens_index)
        # get logits vectors
        tokens_logits = model(**encoded_sentence) 
        #print(tokens_logits)
        #print(tokens_logits['logits'].shape)

        # get the mask-token logit
        mask_tokens_logits = tokens_logits[0][ mask_tokens_index]
        #print(mask_tokens_logits.shape)

        # get the k highest logits
        top_tokens = torch.topk(mask_tokens_logits, 1, dim=1).indices#.tolist()        
        #print(top_tokens)

        # decode the batch of tokens, i.e. get the predicted tokens (each token is represented by an index in the vocabulary)
        predicted_tokens = tokenizer.batch_decode(top_tokens) 
        #print(predicted_tokens)

    return predicted_tokens









size_test = 10000

print(f"Downloading models...")
# select the italian model to test
model = AutoModel.from_pretrained('dbmdz/bert-base-italian-cased').to(device)
tokenizer = AutoTokenizer.from_pretrained('dbmdz/bert-base-italian-cased')



print("Loading paisa sentences...")
sent_neg = load("../Inputs/paisa_sent_neg2.joblib")
sent_pos = load("../Inputs/paisa_sent_pos2.joblib")

print("Tokenization...")
neg_tok = []
for sent in sent_neg:
    tok_sent = tokenizer.tokenize(sent)
    neg_tok.append(tok_sent)

neg_tok_size = []
for tok in neg_tok:
    neg_tok_size.append(len(tok))


pos_tok = []
for sent in sent_pos:
    tok_sent = tokenizer.tokenize(sent)
    pos_tok.append(tok_sent)

pos_tok_size = []
for tok in pos_tok:
    pos_tok_size.append(len(tok))




pos_tok_size_dict = {}
neg_tok_size_dict = {}
'''
print(max(pos_tok_size))
print(min(pos_tok_size))
print(max(neg_tok_size))
print(min(neg_tok_size))
'''


for elem in range(5,57):
    m=0
    for size in pos_tok_size:
        if size == elem:
            m+=1
    pos_tok_size_dict[elem] = m

#print(pos_tok_size_dict)

mn = stat.mean(pos_tok_size)
print(f"\npos len in tok mean {mn}, median {stat.median(pos_tok_size)}, mode {stat.mode(pos_tok_size)}\n\n")



for elem in range(6,75):
    m=0
    for size in neg_tok_size:
        if size == elem:
            m+=1
    if m != 0:
        neg_tok_size_dict[elem] = m
#print(neg_tok_size_dict)

mn = stat.mean(neg_tok_size)
print(f"neg len in tok mean {mn}, median {stat.median(neg_tok_size)}, mode {stat.mode(neg_tok_size)}\n\n")

print("POS")
for key, val in pos_tok_size_dict.items():
    print(f"{key}\t{val}\n")

print("NEG") 
for key, val in neg_tok_size_dict.items():
    print(f"{key}\t{val}\n")


'''

pyplot.bar(list(pos_tok_size_dict.keys()), list(pos_tok_size_dict.values()))
matplotlib.pyplot.axvline(x=mn, linestyle ="--", color = "purple")
matplotlib.pyplot.title("positive")
pyplot.savefig("plot_pos_mn.png")

pyplot.bar(list(neg_tok_size_dict.keys()), list(neg_tok_size_dict.values()))
#matplotlib.pyplot.axvline(x=mn, linestyle ="--", color = "purple")
matplotlib.pyplot.title("negative")
#pyplot.savefig("plot_neg_mn.png")
pyplot.savefig("plot_neg.png")
'''
'''
### log normal distrib
val = [log(n) for n in list(pos_tok_size_dict.keys())]
pyplot.bar(val, list(pos_tok_size_dict.values()))
#matplotlib.pyplot.axvline(x=mn, linestyle ="--", color = "purple")
matplotlib.pyplot.title("positive")
pyplot.savefig("plot_pos_log.png")

'''
'''
val = [log(n) for n in list(neg_tok_size_dict.keys())]
pyplot.bar(val, list(neg_tok_size_dict.values()))
#matplotlib.pyplot.axvline(x=mn, linestyle ="--", color = "purple")
matplotlib.pyplot.title("negative")
pyplot.savefig("plot_neg_log.png")

neg_len = []
pos_len = []

for sent in sent_neg:
    neg_len.append(len(sent))

for sent in sent_pos:
    pos_len.append(len(sent))

pos_len_dict = {}
for elem in range(39, 306):
    m = 0
    for num in pos_len:
        if num == elem:
            m += 1
    if m!=0:
        pos_len_dict[elem] = m

mn = stat.mean(pos_len)
print(f"pos len mean {mn}, median {stat.median(pos_len)}, mode {stat.mode(pos_len)}")

neg_len_dict = {}
for elem in range(39, 393):
    m = 0
    for num in neg_len:
        if num == elem:
            m += 1
    if m != 0:
        neg_len_dict[elem] = m

mn = stat.mean(neg_len)
print(f"neg len mean {mn}, median {stat.median(neg_len)}, mode {stat.mode(neg_len)}")

pyplot.bar(list(neg_len_dict.keys()), list(neg_len_dict.values()))
matplotlib.pyplot.axvline(x=mn, linestyle ="--", color = "purple")
matplotlib.pyplot.title("negative")
pyplot.savefig("plot_neg_len_mn.png")
#pyplot.savefig("plot_neg_len.png")


pyplot.bar(list(pos_len_dict.keys()), list(pos_len_dict.values()))
matplotlib.pyplot.axvline(x=mn, linestyle ="--", color = "purple")
matplotlib.pyplot.title("positive")
pyplot.savefig("plot_pos_len_mn.png")
#pyplot.savefig("plot_pos_len.png")
'''


### log norm distr
'''
val = [log(n) for n in list(neg_len_dict.keys())]
pyplot.bar(val, list(neg_len_dict.values()))
#matplotlib.pyplot.axvline(x=mn, linestyle ="--", color = "purple")
matplotlib.pyplot.title("negative")
pyplot.savefig("plot_neg_len_log.png")


val = [log(n) for n in list(pos_len_dict.keys())]
pyplot.bar(val, list(pos_len_dict.values()))
#matplotlib.pyplot.axvline(x=mn, linestyle ="--", color = "purple")
matplotlib.pyplot.title("positive")
pyplot.savefig("plot_pos_len_log.png")
'''