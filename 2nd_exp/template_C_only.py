# template context only tested with classifier to check if the classifier works better on a single sentence than on two consecutive sentences



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
fprofarray = build_array(fProf_file)
mprofarray = build_array(mProf_file)
professionsarray = {"f": fprofarray, "m": mprofarray}
fnamearray = build_array(fName_file)
mnamearray = build_array(mName_file)
name_arrays = {"f": fnamearray, "m": mnamearray}
pronouns_maj = {"f": "Lei", "m": "Lui"}





# set up list for patterns that, for the CpTp setting, predict for the mask the same verb that was in the context
template_sentences_pos = []

total_sentences = 0 # counts tried sentences
tot_good_preds = 0 # counts sentences with repetition
detail_verbs = {v : 0 for v in list_verbs} # counts, for each verb, how many times it is repeated in the mask if present in context


for gender in ["f", "m"]:
    current_pronouns_maj = pronouns_maj[gender]

    name_arrays_available = name_arrays[gender]
    shuffle(name_arrays_available)
    for name_available in name_arrays_available:
        batch_sentences = [] # batch of sentences to try in this cycle
        batch_verbs = [] # batch of verbs to try in this cycle
        
        professionsarray_available = professionsarray[gender]
        shuffle(professionsarray_available)
        for profession_available in professionsarray_available:
            
            current_list_verbs = list_verbs.copy()
            shuffle(current_list_verbs)

            found = False # to stop when a good verb is found

            for verb_available in current_list_verbs[:50]:
                #print(f"current verb : {verb_available}")
                #if not complete_check and found:
                #    break

                template_Cp = "NOM è PROF che ha l'abitudine di ACT."
                pattern_n = r"NOM"
                pattern_p = r"PROF"
                pattern_a = r"ACT"
                #current_sentence = build_masked_context(name_available, profession_available, verb_available, current_pronouns_maj, mask_token = verb_available)
                current_sentence = re.sub(pattern_n, name_available, template_Cp)
                current_sentence = re.sub(pattern_p, profession_available, current_sentence)
                current_sentence = re.sub(pattern_a, verb_available, current_sentence)
                #print(current_sentence)
                #quit()

                template_sentences_pos.append(current_sentence)
                total_sentences += 1

                
                #if total_sentences % 5000 == 0:
                if total_sentences % 400 == 0:
                    print(f"current : {total_sentences}")
                    print(current_sentence)
                
                if total_sentences > 300:
                    break

'''for gender in ["f", "m"]:
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
'''

print(f"Splitting template sentences in neg and pos...")
# create the CpTp set

'''for pattern in template_sentences_pos:
  # build sentences putting the conjugated verb instead of the mask
  sent = build_masked_context(pattern["name_available"], pattern["profession_available"],
                               pattern["verb"], pattern["current_pronouns_maj"], pattern["masked_prediction"])
  template_sentences_pos.append(sent)
'''
# create the CnTn set
template_sentences_Cn = []

pat_and_repl = [r"che ha","che non ha"]

for sent in template_sentences_pos:
    sent_neg = sent
    sent_Cn = re.sub(pat_and_repl[0], pat_and_repl[1], sent_neg)

    template_sentences_Cn.append(sent_Cn)




print(f"\nCp : {len(template_sentences_pos)}\nCn : {len(template_sentences_Cn)}\n")


#############################
### template set encoding ###
#############################



print(f"Extracting CLS encoding for template sentences...")
# extract CLS for each template sentence
# for each set of sentences, we encode each sentence

all_cls_encodings = []
for templ_list in [template_sentences_Cn, template_sentences_pos]:
  m = 0 
  for sentence in templ_list:
    
    
    sentence_encoded = tokenizer.encode_plus(sentence, padding=True, add_special_tokens=True, return_tensors="pt").to(device)

    # then extract only the outputs for each sentence
    with torch.no_grad():
        tokens_outputs = model(**sentence_encoded )

    # for each set of outputs we only keep the one of the CLS token, namely the first token of each sentence
    embeddings = tokens_outputs[0]
    cls_encodings = embeddings[:,0,:]

  
    m+=1
    cls_encodings = cls_encodings.cpu().numpy()
    if m == 1:
        all_cls_encodings = cls_encodings
    if m > 1:
        all_cls_encodings = np.vstack((all_cls_encodings,cls_encodings))
    if m % 50 == 0:
        print(str(m) + "\textracted")
   
    
   
   
  if templ_list == template_sentences_Cn:
      cls_temp_Cn = all_cls_encodings
  elif templ_list == template_sentences_pos:
      cls_temp_pos = all_cls_encodings
     


np.random.shuffle(cls_temp_Cn)
np.random.shuffle(cls_temp_pos)

cls_temp_pos = cls_temp_pos[:size_test]
cls_temp_Cn = cls_temp_Cn[:size_test]






############################
### masked template test ###
############################




#train_temp = np.concatenate((cls_encodings_pos[:train_size], cls_encodings_neg[:train_size]))
#train_temp_lab = np.concatenate((np.zeros(train_size), np.ones(train_size)))
#test_temp = np.concatenate((cls_encodings_pos[train_size:], cls_encodings_neg[train_size:]))
#test_temp_lab = np.concatenate((np.zeros(test_size), np.ones(test_size)))



test_temp_pos = cls_temp_pos[:size_test]
test_temp_Cn = cls_temp_Cn[:size_test]

test_temp_lab_pos = np.zeros(size_test)
test_temp_lab_Cn = np.ones(size_test)




#data normalization
scaler = load(f"../Inputs/scaler.joblib")
test_2_pos = scaler.transform(test_temp_pos)
test_2_Cn = scaler.transform(test_temp_Cn)





#########################################
### classification with the MLP model ###
#########################################


print("Testing with MLP classifiers...")

template_result_pos = []
template_result_Cn = []

for n in range(1, 13):
   clf = load(f"../Inputs/non_classifier_{n}.joblib")
   
for n in range(1, 13):
   clf = load(f"../Inputs/non_classifier_{n}.joblib")
   
   
   predicted = clf.predict(test_2_pos)
   right_pred = clf.score(test_2_pos, test_temp_lab_pos)
   if len(confusion_matrix(test_temp_lab_pos, predicted).ravel()) == 4:
       tn, fp, fn, tp = confusion_matrix(test_temp_lab_pos, predicted).ravel()
       template_result_pos.append(f"Score\t{right_pred}\n\nTrue neg\t{tn}\nFalse pos\t{fp}\nFalse neg\t{fn}\nTrue pos\t{tp}\n\n")
   else:
       conf_matr = confusion_matrix(test_temp_lab_pos, predicted).ravel()
       template_result_pos.append(f"Score\t{right_pred}\n\nConfusion matrix : {conf_matr}\n\n")
   #template_result.append(f"Method\t{solv}\nNb hidden layers\t{str(hl)}\nAlpha\t{str(a)}\nScores\t{right_pred}\n\nTrue neg\t{tn}\nFalse pos\t{fp}\nFalse neg\t{fn}\nTrue pos\t{tp}\n\n")



   predicted = clf.predict(test_2_Cn)
   right_pred = clf.score(test_2_Cn, test_temp_lab_Cn)
   if len(confusion_matrix(test_temp_lab_Cn, predicted).ravel()) == 4:
       tn, fp, fn, tp = confusion_matrix(test_temp_lab_Cn, predicted).ravel()
       template_result_Cn.append(f"Score\t{right_pred}\n\nTrue neg\t{tn}\nFalse pos\t{fp}\nFalse neg\t{fn}\nTrue pos\t{tp}\n\n")
   else:
       conf_matr = confusion_matrix(test_temp_lab_Cn, predicted).ravel()
       template_result_Cn.append(f"Score\t{right_pred}\n\nConfusion matrix : {conf_matr}\n\n")

'''   
   predicted = clf.predict(test_2_pos)
   right_pred = clf.score(test_2_pos, test_temp_lab_pos)
   tn, fp, fn, tp = confusion_matrix(test_temp_lab_pos, predicted).ravel()
   #template_result.append(f"Method\t{solv}\nNb hidden layers\t{str(hl)}\nAlpha\t{str(a)}\nScores\t{right_pred}\n\nTrue neg\t{tn}\nFalse pos\t{fp}\nFalse neg\t{fn}\nTrue pos\t{tp}\n\n")
   template_result_pos.append(f"Score\t{right_pred}\n\nTrue neg\t{tn}\nFalse pos\t{fp}\nFalse neg\t{fn}\nTrue pos\t{tp}\n\n")

   predicted = clf.predict(test_2_CnTn)
   right_pred = clf.score(test_2_CnTn, test_temp_lab_CnTn)
   tn, fp, fn, tp = confusion_matrix(test_temp_lab_CnTn, predicted).ravel()
   #template_result.append(f"Method\t{solv}\nNb hidden layers\t{str(hl)}\nAlpha\t{str(a)}\nScores\t{right_pred}\n\nTrue neg\t{tn}\nFalse pos\t{fp}\nFalse neg\t{fn}\nTrue pos\t{tp}\n\n")
   template_result_CnTn.append(f"Score\t{right_pred}\n\nTrue neg\t{tn}\nFalse pos\t{fp}\nFalse neg\t{fn}\nTrue pos\t{tp}\n\n")

   predicted = clf.predict(test_2_CnTp)
   right_pred = clf.score(test_2_CnTp, test_temp_lab_CnTp)
   tn, fp, fn, tp = confusion_matrix(test_temp_lab_CnTp, predicted).ravel()
   #template_result.append(f"Method\t{solv}\nNb hidden layers\t{str(hl)}\nAlpha\t{str(a)}\nScores\t{right_pred}\n\nTrue neg\t{tn}\nFalse pos\t{fp}\nFalse neg\t{fn}\nTrue pos\t{tp}\n\n")
   template_result_CnTp.append(f"Score\t{right_pred}\n\nTrue neg\t{tn}\nFalse pos\t{fp}\nFalse neg\t{fn}\nTrue pos\t{tp}\n\n")

   predicted = clf.predict(test_2_CpTn)
   right_pred = clf.score(test_2_CpTn, test_temp_lab_CpTn)
   tn, fp, fn, tp = confusion_matrix(test_temp_lab_CpTn, predicted).ravel()
   #template_result.append(f"Method\t{solv}\nNb hidden layers\t{str(hl)}\nAlpha\t{str(a)}\nScores\t{right_pred}\n\nTrue neg\t{tn}\nFalse pos\t{fp}\nFalse neg\t{fn}\nTrue pos\t{tp}\n\n")
   template_result_CpTn.append(f"Score\t{right_pred}\n\nTrue neg\t{tn}\nFalse pos\t{fp}\nFalse neg\t{fn}\nTrue pos\t{tp}\n\n")

'''



print("TEMPLATE TEST POS\n\n")
for scores in template_result_pos:
   print(scores)


print("TEMPLATE TEST Cn\n\n")
for scores in template_result_Cn:
   print(scores)
