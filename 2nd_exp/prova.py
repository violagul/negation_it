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









size_test = 200

print(f"Downloading models...")
# select the italian model to test
tokenizer = AutoTokenizer.from_pretrained('dbmdz/bert-base-italian-cased')
model_mask = AutoModelForMaskedLM.from_pretrained('dbmdz/bert-base-italian-cased').to(device)
model = AutoModel.from_pretrained('dbmdz/bert-base-italian-cased').to(device)



###########################
### masked template set ###
###########################






print(f"Building template sentences...")
# load names, professions and verbs for the templates
path = r"../Inputs"
fName_file_path = f"{path}/100_names_f.txt"
mName_file_path = f"{path}/100_names_m.txt"
fProf_file_path = f"{path}/100_mestieri_f.txt"
mProf_file_path = f"{path}/100_mestieri_m.txt"
hypo_file_path = f"{path}/frasi_it.txt"

fName_file = open(fName_file_path, "r")
mName_file = open(mName_file_path, "r")
fProf_file = open(fProf_file_path, "r")
mProf_file = open(mProf_file_path, "r")

list_verbs = load(f"{path}/base_verbs.joblib")





# dictionaries of names, professions and pronouns indexed by gender for template construction
#professionsarray = {"f": build_array(fProf_file)[:10], "m": build_array(mProf_file)[10]} 
# buildarray is a function for creating lists from txt files        
fprofarray = build_array(fProf_file)[10]
mprofarray = build_array(mProf_file)[10]
professionsarray = {"f": fprofarray, "m": mprofarray}
fnamearray = build_array(fName_file)[20]
mnamearray = build_array(mName_file)[20]
name_arrays = {"f": fnamearray, "m": mnamearray}
pronouns_maj = {"f": "Lei", "m": "Lui"}





# set up list for patterns that, for the CpTp setting, predict for the mask the same verb that was in the context
list_good_patterns_model = []

total_sentences = 0 # counts tried sentences
tot_good_preds = 0 # counts sentences with repetition
detail_verbs = {v : 0 for v in list_verbs} # counts, for each verb, how many times it is repeated in the mask if present in context


size_batches = 8

for gender in ["f", "m"]:
    current_pronouns_maj = pronouns_maj[gender]

    for name_available in name_arrays[gender]:
        batch_sentences = [] # batch of sentences to try in this cycle
        batch_verbs = [] # batch of verbs to try in this cycle
        
        for profession_available in professionsarray[gender]:
            
            current_list_verbs = list_verbs.copy()
            shuffle(current_list_verbs)

            found = False # to stop when a good verb is found

            for verb_available in current_list_verbs:
                #print(f"current verb : {verb_available}")
                #if not complete_check and found:
                #    break

                
                current_sentence = build_masked_context(name_available, profession_available, verb_available, current_pronouns_maj, mask_token = tokenizer.mask_token)

                #print(current_sentence)
                #quit()

                batch_sentences.append(current_sentence)
                batch_verbs.append(verb_available)
                total_sentences += 1

                
                if total_sentences % 5000 == 0:
                    print(f"current : {total_sentences}, found : {len(list_good_patterns_model)}")

                # get the result at the end of the batch
                if len(batch_sentences) == size_batches:
                    new_sentence, found, nb_good_pred, found_verbs = make_and_encode_batch(batch_sentences, tokenizer, model_mask, device, batch_verbs, name_available, profession_available, current_pronouns_maj, found) 
                    tot_good_preds+=nb_good_pred
                    if new_sentence!= None:
                        list_good_patterns_model.append(new_sentence)
                    batch_sentences = []
                    batch_verbs = []
                    for found_verb in found_verbs:
                        detail_verbs[found_verb] +=  1 # add one repetition to the count for the found verb


            # repetition for what is left out of the last batch
            if len(batch_sentences) > 0: 
                new_sentence, found, nb_good_pred, found_verbs = make_and_encode_batch(batch_sentences, tokenizer, model_mask, device, batch_verbs, name_available, profession_available, current_pronouns_maj, found)

                tot_good_preds += nb_good_pred
                if new_sentence != None:
                    list_good_patterns_model.append(new_sentence)
                batch_sentences = []
                batch_verbs = []
                for found_verb in found_verbs:
                    detail_verbs[found_verb] += 1


print(f"Splitting template sentences in neg and pos...")
# create the CpTp set
template_sentences_pos =[]
for pattern in list_good_patterns_model:
  # build sentences putting the conjugated verb instead of the mask
  sent = build_masked_context(pattern["name_available"], pattern["profession_available"],
                               pattern["verb"], pattern["current_pronouns_maj"], pattern["masked_prediction"])
  template_sentences_pos.append(sent)

# create the CnTn set
template_sentences_neg = []
pat_and_repl = [[r"che ha","che non ha"],[r" Lei ", " Lei non "],[r" Lui "," Lui non "]]

for sent in template_sentences_pos:
  sent_neg = sent
  for pair in pat_and_repl:
    sent_neg = re.sub(pair[0], pair[1], sent_neg)
  template_sentences_neg.append(sent_neg)
  

#############################
### template set encoding ###
#############################



print(f"Extracting CLS encoding for template sentences...")
# extract CLS for each template sentence
# for each set of sentences, we encode each sentence


all_sent_list = template_sentences_neg
all_sent_list.extend(template_sentences_pos)
print(type(all_sent_list))
print(all_sent_list)
for sent_list in all_sent_list:
  batch_encoded = tokenizer.batch_encode_plus(sent_list, padding=True, add_special_tokens=True, return_tensors="pt").to(device)

  # then extract only the outputs for each sentence
  with torch.no_grad():
    tokens_outputs = model(**batch_encoded )

  # for each set of outputs we only keep the one of the CLS token, namely the first token of each sentence
  embeddings = tokens_outputs[0]
  cls_encodings = embeddings[:,0,:]

  print(cls_encodings)

  cls_encodings = cls_encodings.cpu().numpy()

  if sent_list == template_sentences_neg:
    cls_temp_neg = cls_encodings
  elif sent_list == template_sentences_pos:
    cls_temp_pos = cls_encodings


np.random.shuffle(cls_temp_neg)
np.random.shuffle(cls_temp_pos)

cls_temp_pos = cls_temp_pos[:size_test]
cls_temp_neg = cls_temp_neg[:size_test]







############################
### masked template test ###
############################




#train_temp = np.concatenate((cls_encodings_pos[:train_size], cls_encodings_neg[:train_size]))
#train_temp_lab = np.concatenate((np.zeros(train_size), np.ones(train_size)))
#test_temp = np.concatenate((cls_encodings_pos[train_size:], cls_encodings_neg[train_size:]))
#test_temp_lab = np.concatenate((np.zeros(test_size), np.ones(test_size)))



test_temp = np.concatenate((cls_temp_pos[:size_test], cls_temp_neg[:size_test]))
test_temp_lab = np.concatenate((np.zeros(size_test), np.ones(size_test)))



#scaler.fit(train_temp)
#train = scaler.transform(train_temp)
#test_2 = scaler.transform(test_temp)


#data normalization
scaler = StandardScaler()
scaler.fit(test_temp)
test_2 = scaler.transform(test_temp)





#########################################
### classification with the MLP model ###
#########################################


print("Testing with MLP classifiers...")

template_result = []

for n in range(1, 13):
   clf = load(f"../Inputs/non_classifier_{n}.joblib")
   
   
   predicted = clf.predict(test_2)
   right_pred = clf.score(test_2, test_temp_lab)
   tn, fp, fn, tp = confusion_matrix(test_temp_lab, predicted).ravel()
   template_result.append(f"Method: {solv}\tNb hidden layers: {str(hl)}\tAlpha: {str(a)}\n {right_pred}%\n\nTrue neg = {tn}\nFalse pos = {fp}\nFalse neg = {fn}\nTrue pos = {tp}")





print("TEMPLATE TEST\n\n")
for scores in template_result:
   print(scores)