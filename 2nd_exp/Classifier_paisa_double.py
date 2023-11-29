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

for s in sent:
  found = re.findall(negC_patt, s)
  for elem in found:
    double = re.search(negT_patt, elem)
    if not double: # exclude couples of sentences where both are negative
      CnTp.append(elem)
  found_2 = re.findall(negT_patt, s)
  for elem in found_2:
    double2 = re.search(negC_patt, elem)
    if not double2:
      CpTn.append(elem)





################################
### CnTp - CpTn set encoding ###
################################



size_batches = 8
print(f"Extracting the CLS encodings from CpTn/CnTp sentences from PAISA...")
# encode the CnTp ad CpTn sentences
for sent_list in [CpTn, CnTp]:
  for sent in sent_list:
    batch_sent.append(sent)
    if len(batch_sent) == size_batches:
      batch_encoded = tokenizer.batch_encode_plus(batch_sent, padding=True, add_special_tokens=True, return_tensors="pt").to(device)
      # then extract only the outputs for each sentence
      with torch.no_grad():
        tokens_outputs = model(**batch_encoded )

      # for each set of outputs we only keep the one of the CLS token, namely the first token of each sentence
      embeddings = tokens_outputs[0]
      batch_cls = embeddings[:,0,:]
      for elem in batch_cls:
        cls_encodings.append(elem)
      batch_sent = []

  if len(batch_sent) > 0:
    batch_encoded = tokenizer.batch_encode_plus(batch_sent, padding=True, add_special_tokens=True, return_tensors="pt").to(device)
    # then extract only the outputs for each sentence
    with torch.no_grad():
      tokens_outputs = model(**batch_encoded )

    # for each set of outputs we only keep the one of the CLS token, namely the first token of each sentence
    embeddings = tokens_outputs[0]
    batch_cls = embeddings[:,0,:]
    for elem in batch_cls:
      cls_encodings.append(elem)
    batch_sent = []

  cls_encodings = cls_encodings.cpu().numpy()

  if sent_list == CnTp:
    cls_CnTp = cls_encodings
  elif sent_list == CpTn:
    cls_CpTn = cls_encodings


np.random.shuffle(cls_CnTp)
np.random.shuffle(cls_CpTn)

cls_CpTn = cls_CpTn[:size_test]
cls_CnTp = cls_CnTp[:size_test]




############################
### CpTn - CnTp set test ###
############################




test_CnTp = np.array(cls_CnTp)
test_CpTn = np.array(cls_CpTn)
test_CnTp_lab = np.array(np.ones(size_test))
test_CpTn_lab = np.array(np.ones(size_test))


# data normalization
scaler = StandardScaler()
scaler.fit(test_CnTp)
test_3 = scaler.transform(test_CnTp)
scaler.fit(test_CpTn)
test_4 = scaler.transform(test_CpTn)









#########################################
### classification with the MLP model ###
#########################################




print("Testing with MLP classifiers...")

CnTp_result = []
CpTn_result = []
for n in range(1, 13):
   clf = load(f"../Inputs/non_classifier_{n}.joblib")

   predicted = clf.predict(test_3)
   right_pred = clf.score(test_3, test_CnTp_lab)
   tn, fp, fn, tp = confusion_matrix(test_CnTp_lab, predicted).ravel()
   CnTp_result.append(f"Method: {solv}\tNb hidden layers: {str(hl)}\tAlpha: {str(a)}\n {right_pred}%\n\nTrue neg = {tn}\nFalse pos = {fp}\nFalse neg = {fn}\nTrue pos = {tp}")

   predicted = clf.predict(test_4)
   right_pred = clf.score(test_4, test_CpTn_lab)
   tn, fp, fn, tp = confusion_matrix(test_CpTn_lab, predicted).ravel()
   CpTn_result.append(f"Method: {solv}\tNb hidden layers: {str(hl)}\tAlpha: {str(a)}\n {right_pred}%\n\nTrue neg = {tn}\nFalse pos = {fp}\nFalse neg = {fn}\nTrue pos = {tp}")







print("PAISA CnTp TEST\n\n")
for scores in CnTp_result:
   print(scores)

print("PAISA CpTn TEST\n\n")
for scores in CpTn_result:
   print(scores)