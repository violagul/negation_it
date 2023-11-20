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










##############################
### paisa "non" extraction ###
##############################









###size_test = 10000
size_test = 1000

print(f"Downloading models...")
# select the italian model to test
model = AutoModel.from_pretrained('dbmdz/bert-base-italian-cased').to(device)
tokenizer = AutoTokenizer.from_pretrained('dbmdz/bert-base-italian-cased')


# upload the Italian corpus
with open(r"../data/paisa.raw.utf8", encoding='utf8') as infile:
    paisa = infile.read()

print(f"Extracting Wikipedia texts from PAISA...")
# from the corpus, select all texts containing "wiki" in their tag's url
wiki_pattern = r"<text.*wiki.*(?:\n.*)+?\n</text>\n" 
paisa_wiki = re.findall(wiki_pattern, paisa)
#print(f"Number of texts from a site containing 'wiki' in their URL: {len(paisa_wiki)}")
#paisa_wiki = paisa


print(f"Extracting sentences from PAISA...")
# pattern for finding whole sentences in the texts (defined by the capital letter in the beginning, the period at the end and a minimum length)
sent = []
pattern = r" [A-Z][a-z ]*[,:]?[a-z ]+[,:]?[a-z ][,:]?[a-z]+\. \b"  # finds kind of acceptable sentences

for text in paisa_wiki:
  found = re.findall(pattern, text)
  for elem in  found:
    if len(elem) > 25:
      sent.append(elem)
  if len(sent)> size_test*100:
    break

#print(f"Number of sentences: {len(sent)}")


print(f"Extracting negative sentences from PAISA...")
# splitting the sentences above into two lists:
sent_pos = []
sent_neg = []

# pattern to find the negation in a sentence
neg_patt = r"\b[Nn]on\b"  

for s in sent:
  if len(sent_neg) > 20 and len(sent_pos)>20:
     break
  matches = re.search(neg_patt, s)
  if matches:
    sent_neg.append(s)
  else:
    sent_pos.append(s)




size_test = min(size_test, len(sent_neg), len(sent_pos))

shuffle(sent_neg)
shuffle(sent_pos)



# select a fixed numb of sentences to test
sent_neg = sent_neg[:size_test]
sent_pos = sent_pos[:size_test]

print(f"Extracting CLS ecodings for PAISA sentences...")
### extract CLS
# for each set of sentences, we encode each sentence
for sent_list in [sent_neg, sent_pos]:
  batch_encoded = tokenizer.batch_encode_plus(sent_list, padding=True, add_special_tokens=True, return_tensors="pt").to(device)

  # then extract only the outputs for each sentence
  with torch.no_grad():
    tokens_outputs = model(**batch_encoded)

  # for each set of outputs we only keep the one of the CLS token, namely the first token of each sentence
  #print(tokens_outputs)
  embeddings = tokens_outputs[0]
  cls_encodings = embeddings[:, 0, :]
  #print(cls_encodings.shape)
  cls_encodings = cls_encodings.cpu().numpy()
  

  if sent_list == sent_neg:
    cls_encodings_neg = cls_encodings
  elif sent_list == sent_pos:
    cls_encodings_pos = cls_encodings


print(f"Training MLP...")
#train = torch.zeros(cls_encodings_neg.shape[0]*2, cls_encodings_neg.shape[1])
#train[cls_encodings_neg.shape[0]] = cls_encodings_neg[:9000]
#train = train.append(cls_encodings_pos[:9000])

# we use 90% of data as training and 10% as test
train_size = round(size_test*0.9)
train = np.concatenate((cls_encodings_pos[:train_size], cls_encodings_neg[:train_size]), 0) # shape num_sent x 768
labels = np.concatenate((np.zeros(train_size), np.ones(train_size)))
test = np.concatenate((cls_encodings_pos[train_size:], cls_encodings_neg[train_size:]), 0)
test_size = int(size_test - train_size)
test_lab = np.concatenate((np.zeros(test_size), np.ones(test_size)))


# data normalization
scaler = StandardScaler()
scaler.fit(train)
dati_scaled = scaler.transform(train)


X = dati_scaled 
test = scaler.transform(test)
#print(test)
#print(test_lab)

y = labels
    
X, y = skshuffle(X, y, random_state=42)







###########################
### masked template set ###
###########################






print(f"Building template sentences...")

model_mask = AutoModelForMaskedLM.from_pretrained('dbmdz/bert-base-italian-cased').to(device)




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

list_verbs = load(f"{path}/base_verbs.joblib")[:10]





# dictionaries of names, professions and pronouns indexed by gender for template construction
#professionsarray = {"f": build_array(fProf_file)[:10], "m": build_array(mProf_file)[10]} 
# buildarray is a function for creating lists from txt files        
fprofarray = build_array(fProf_file)[:10]
mprofarray = build_array(mProf_file)[:10]
for elem in mprofarray:
   print(elem)
professionsarray = {"f": fprofarray, "m": mprofarray}
fnamearray = build_array(fName_file)[:10]
mnamearray = build_array(mName_file)[:10]
name_arrays = {"f": fnamearray, "m": mnamearray}
pronouns_maj = {"f": "Lei", "m": "Lui"}





# set up list for patterns that, for the CpTp setting, predict for the mask the same verb that was in the context
list_good_patterns_model = []

total_sentences = 0 # counts tried sentences
tot_good_preds = 0 # counts sentences with repetition
detail_verbs = {v : 0 for v in list_verbs} # counts, for each verb, how many times it is repeated in the mask if present in context


size_batches = 100

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

                #if total_sentences % 1000 == 0:
                if total_sentences % 100 == 0:
                    print(f"current : {total_sentences}, {len(list_good_patterns_model)}, {current_sentence}")

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
for sent_list in [template_sentences_neg, template_sentences_pos]:
  batch_encoded = tokenizer.batch_encode_plus(sent_list, padding=True, add_special_tokens=True, return_tensors="pt").to(device)

  # then extract only the outputs for each sentence
  with torch.no_grad():
    tokens_outputs = model(**batch_encoded )

  # for each set of outputs we only keep the one of the CLS token, namely the first token of each sentence
  cls_encodings = tokens_outputs.last_hidden_state[:, 0, :]

  cls_encodings = cls_encodings.cpu().numpy()

  if sent_list == template_sentences_neg:
    cls_temp_neg = cls_encodings
  elif sent_list == template_sentences_pos:
    cls_temp_pos = cls_encodings


cls_temp_neg.shuffle()
cls_temp_pos.shuffle()

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



scaler.fit(test_temp)
test_2 = scaler.transform(test_temp)









########################################
###CnTp and CpTn sentences from paisa###
########################################


# pattern for couples of sentences
double_sent = r"(?<= )[A-Z][a-z ]*[,:]?[a-z ]+[,:]?[a-z ][,:]?[a-z]+\. [A-Z][a-z ]*[,:]?[a-z ]+[,:]?[a-z ][,:]?[a-z]+\.(?= \b)"


# patterns for "non" in context: in the first of two sentences or the second of two sentences
negC_patt = r".*[Nn]on.*\..*\."
negT_patt = r".*\..*[Nn]on.*\." 


# extract couples of sentences
sent = []
for text in paisa_wiki:
  found = re.findall(double_sent, text)
  for elem in found:
    if len(elem)>25:
      sent.append(elem)

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
  for elem in found:
    double2 = re.search(negC_patt, elem)
    if not double2:
      CpTn.append(elem)





################################
### CnTp - CpTn set encoding ###
################################





# encode the CnTp ad CpTn sentences
for sent_list in [CpTn, CnTp]:
  batch_encoded = tokenizer.batch_encode_plus(sent_list, padding=True, add_special_tokens=True, return_tensors="pt").to(device)

  # then extract only the outputs for each sentence
  with torch.no_grad():
    tokens_outputs = model(**batch_encoded )

  # for each set of outputs we only keep the one of the CLS token, namely the first token of each sentence
  cls_encodings = tokens_outputs.last_hidden_state[:, 0, :]

  cls_encodings = cls_encodings.cpu().numpy()

  if sent_list == CnTp:
    cls_CnTp = cls_encodings
  elif sent_list == CpTn:
    cls_CpTn = cls_encodings


cls_CnTp.shuffle()
cls_CpTn.shuffle()

cls_CpTn = cls_CpTn[:size_test]
cls_CnTp = cls_CnTp[:size_test]




############################
### CpTn - CnTp set test ###
############################




test_CnTp = np.array(cls_CnTp)
test_CpTn = np.array(cls_CpTn)
test_CnTp_lab = np.array(np.ones(size_test))
test_CpTn_lab = np.array(np.ones(size_test))


scaler.fit(test_CnTp)
test_3 = scaler.transform(test_CnTp)
scaler.fit(test_CpTn)
test_4 = scaler.transform(test_CpTn)










########################################
### classifier creation and training ###
########################################





paisa_result =[]
template_result = []
CnTp_result = []
CpTn_result = []


# set up the MLP classifier
# solver : adam or sgd
# hidden_layer_sizes : 40,40 or 350,350
# alpha : between 1e-5 and 1e-2
for hl in [(40,40), (350,350)]:
  for a in [1e-2, 1e-3, 1e-4, 1e-5]:
    for solv in ["adam", "sgd"]:
      clf = MLPClassifier(solver = "adam", alpha = a,
                    hidden_layer_sizes=hl, random_state = 1)

      # train on data
      clf.fit(X, y)

      # see predictions on the dataset
      predicted = clf.predict(test)
      right_pred = clf.score(test, test_lab)
      tn, fp, fn, tp = confusion_matrix(test_lab, predicted).ravel()
      paisa_result.append(f"Method: {solv}\tNb hidden layers: {str(hl)}\tAlpha: {str(a)}\n {right_pred}%\n\nTrue neg = {tn}\nFalse pos = {fp}\nFalse neg = {fn}\nTrue pos = {tp}")

      predicted = clf.predict(test_2)
      right_pred = clf.score(test_2, test_temp_lab)
      tn, fp, fn, tp = confusion_matrix(test_temp_lab, predicted).ravel()
      template_result.append(f"Method: {solv}\tNb hidden layers: {str(hl)}\tAlpha: {str(a)}\n {right_pred}%\n\nTrue neg = {tn}\nFalse pos = {fp}\nFalse neg = {fn}\nTrue pos = {tp}")

      predicted = clf.predict(test_3)
      right_pred = clf.score(test_3, test_CnTp_lab)
      tn, fp, fn, tp = confusion_matrix(test_CnTp_lab, predicted).ravel()
      CnTp_result.append(f"Method: {solv}\tNb hidden layers: {str(hl)}\tAlpha: {str(a)}\n {right_pred}%\n\nTrue neg = {tn}\nFalse pos = {fp}\nFalse neg = {fn}\nTrue pos = {tp}")

      predicted = clf.predict(test_4)
      right_pred = clf.score(test_4, test_CpTn_lab)
      tn, fp, fn, tp = confusion_matrix(test_CpTn_lab, predicted).ravel()
      CpTn_result.append(f"Method: {solv}\tNb hidden layers: {str(hl)}\tAlpha: {str(a)}\n {right_pred}%\n\nTrue neg = {tn}\nFalse pos = {fp}\nFalse neg = {fn}\nTrue pos = {tp}")







print("PAISA' TEST\n\n")
for scores in paisa_result:
   print(scores)

print("TEMPLATE TEST\n\n")
for scores in template_result:
   print(scores)

print("CnTp TEST\n\n")
for scores in CnTp_result:
   print(scores)

print("CpTn TEST\n\n")
for scores in CpTn_result:
   print(scores)




