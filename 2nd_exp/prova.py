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

#from sklearn.model_selection import train_test_split

from tools.build_array import build_array, build_hypo, build_masked_context
from tools.chech_conjug import check_conjugation

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





print("Loading paisa_wiki texts...")
paisa_wiki = load("../Inputs/paisa_wiki.joblib")







########################################
###CnTp and CpTn sentences from paisa###
########################################


print(f"Extracting couples of consecutive sentences from PAISA...")
# pattern for couples of sentences
double_sent = r"(?<= )[A-Z][a-z ]*[,:]?[a-z ]+[,:]?[a-z ][,:]?[a-z]+\. [A-Z][a-z ]*[,:]?[a-z ]+[,:]?[a-z ][,:]?[a-z]+\.(?= \b)"


# patterns for "non" in context: in the first of two sentences or the second of two sentences
negC_patt = r".*[Nn]on.*\..*\."
negT_patt = r".*\..*[Nn]on.*\." 


# extract couples of sentences
sent = []
num = 0
for text in paisa_wiki:
  num+=1
  found = re.findall(double_sent, text)
  for elem in found:
    if len(elem)>25:
      sent.append(elem)
  if num % 1000 == 0:
     print(f"{num} of {len(paisa_wiki)} texts analysed, found : {len(sent)}")
     



print(f"Extracting CnTp and CpTn types of sentences from PAISA...")
# create two lists to store: 
CnTp = [] # couples of sentences the first of which is negative
CpTn = [] # couples of sentences the second of which is negative
CnTn = []

for s in sent:
  found = re.findall(negC_patt, s)
  for elem in found:
    double = re.search(negT_patt, elem)
    if not double: # exclude couples of sentences where both are negative
      CnTp.append(elem)
    if double:
      CnTn.append(elem)
  found_2 = re.findall(negT_patt, s)
  for elem in found_2:
    double2 = re.search(negC_patt, elem)
    if not double2:
      CpTn.append(elem)


CpTp = sent
print(f"len CpTn = {len(CpTn)}")
print(f"len CnTp = {len(CnTp)}")
print(f"len CnTn = {len(CnTn)}")
print(f"len CpTp = {len(CpTp)}")



################################
### CnTp - CpTn set encoding ###
################################



print(f"Extracting the CLS encodings from CpTn/CnTp sentences from PAISA...")
# encode the CnTp ad CpTn sentences
all_cls_encodings = []
for sent_list in [CpTn, CnTp, CpTp, CnTn]:
  m = 0
  for sent in sent_list:
    sent_encoded = tokenizer.encode_plus(sent, padding=True, add_special_tokens=True, return_tensors="pt").to(device)
    if m == 0:
      print(sent)


    # then extract only the outputs for each sentence
    with torch.no_grad():
      tokens_outputs = model(**sent_encoded )

    # for each set of outputs we only keep the one of the CLS token, namely the first token of each sentence
    embeddings = tokens_outputs[0]
    cls_encodings = embeddings[:,0,:]
    
    m+=1
    cls_encodings = cls_encodings.cpu().numpy()
    if m == 1:
      all_cls_encodings = cls_encodings
      print(cls_encodings.shape)
    if m > 1:
      all_cls_encodings = np.vstack((all_cls_encodings, cls_encodings))
    if m % 200 == 0:
      print(f"Encoded : {m}")
    

  if sent_list == CnTp:
    cls_CnTp = all_cls_encodings
  elif sent_list == CpTn:
    cls_CpTn = all_cls_encodings
  if sent_list == CnTn:
    cls_CnTn = all_cls_encodings
  if sent_list == CpTp:
    cls_CpTp = all_cls_encodings


np.random.shuffle(cls_CnTp)
np.random.shuffle(cls_CpTn)
np.random.shuffle(cls_CnTn)
np.random.shuffle(cls_CpTn)


size_test_CpTn = min(size_test, len(cls_CpTn))
size_test_CnTp = min(size_test, len(cls_CnTp))
size_test_CnTn = min(size_test, len(cls_CnTn))
size_test_CpTp = min(size_test, len(cls_CpTp))
cls_CpTn = cls_CpTn[:size_test_CpTn]
cls_CnTp = cls_CnTp[:size_test_CnTp]
cls_CpTp = cls_CpTp[:size_test_CpTp]
cls_CnTn = cls_CnTn[:size_test_CnTn]




############################
### CpTn - CnTp set test ###
############################




test_CnTp = np.array(cls_CnTp)
test_CpTn = np.array(cls_CpTn)
test_CnTn = np.array(cls_CnTn)
test_CpTp = np.array(cls_CpTp)
test_CnTp_lab = np.array(np.ones(size_test_CnTp))
test_CpTn_lab = np.array(np.ones(size_test_CpTn))
test_CnTn_lab = np.array(np.ones(size_test_CnTn))
test_CpTp_lab = np.array(np.zeros(size_test_CpTp))


# data normalization
scaler = load(f"../Inputs/scaler.joblib")
test_3 = scaler.transform(test_CnTp)
test_4 = scaler.transform(test_CpTn)
test_5 = scaler.transform(test_CnTn)
test_6 = scaler.transform(test_CpTp)









#########################################
### classification with the MLP model ###
#########################################




print("Testing with MLP classifiers...")

CnTp_result = []
CpTn_result = []
CnTn_result = []
CpTp_result = []

for n in range(1, 13):
   clf = load(f"../Inputs/non_classifier_{n}.joblib")

   predicted = clf.predict(test_3)
   right_pred = clf.score(test_3, test_CnTp_lab)
   tn, fp, fn, tp = confusion_matrix(test_CnTp_lab, predicted).ravel()
   #CnTp_result.append(f"Method\t{solv}\nNb hidden layers\t{str(hl)}\nAlpha\t{str(a)}\nScore\t{right_pred}%\n\nTrue neg\t{tn}\nFalse pos\t{fp}\nFalse neg\t{fn}\nTrue pos\t{tp}\n\n")
   CnTp_result.append(f"Score\t{right_pred}%\n\nTrue neg\t{tn}\nFalse pos\t{fp}\nFalse neg\t{fn}\nTrue pos\t{tp}\n\n")

   predicted = clf.predict(test_4)
   right_pred = clf.score(test_4, test_CpTn_lab)
   tn, fp, fn, tp = confusion_matrix(test_CpTn_lab, predicted).ravel()
   #CpTn_result.append(f"Method\t{solv}\nNb hidden layers\t{str(hl)}\nAlpha\t{str(a)}\nScore\t{right_pred}\n\nTrue neg\t{tn}\nFalse pos\t{fp}\nFalse neg\t{fn}\nTrue pos\t{tp}\n\n")
   CpTn_result.append(f"Score\t{right_pred}%\n\nTrue neg\t{tn}\nFalse pos\t{fp}\nFalse neg\t{fn}\nTrue pos\t{tp}\n\n")
   
   predicted = clf.predict(test_5)
   right_pred = clf.score(test_5, test_CnTn_lab)
   tn, fp, fn, tp = confusion_matrix(test_CnTn_lab, predicted).ravel()
   #CnTp_result.append(f"Method\t{solv}\nNb hidden layers\t{str(hl)}\nAlpha\t{str(a)}\nScore\t{right_pred}%\n\nTrue neg\t{tn}\nFalse pos\t{fp}\nFalse neg\t{fn}\nTrue pos\t{tp}\n\n")
   CnTn_result.append(f"Score\t{right_pred}%\n\nTrue neg\t{tn}\nFalse pos\t{fp}\nFalse neg\t{fn}\nTrue pos\t{tp}\n\n")

   predicted = clf.predict(test_6)
   right_pred = clf.score(test_6, test_CpTp_lab)
   tn, fp, fn, tp = confusion_matrix(test_CpTp_lab, predicted).ravel()
   #CpTn_result.append(f"Method\t{solv}\nNb hidden layers\t{str(hl)}\nAlpha\t{str(a)}\nScore\t{right_pred}\n\nTrue neg\t{tn}\nFalse pos\t{fp}\nFalse neg\t{fn}\nTrue pos\t{tp}\n\n")
   CpTp_result.append(f"Score\t{right_pred}%\n\nTrue neg\t{tn}\nFalse pos\t{fp}\nFalse neg\t{fn}\nTrue pos\t{tp}\n\n")






'''
print("PAISA CnTp TEST\n\n")
for scores in CnTp_result:
   print(scores)

print("PAISA CpTn TEST\n\n")
for scores in CpTn_result:
   print(scores)
'''


print("PAISA CnTn TEST\n\n")
for scores in CnTn_result:
   print(scores)

print("PAISA CpTp TEST\n\n")
for scores in CpTp_result:
   print(scores)