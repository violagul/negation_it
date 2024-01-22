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







##############################
### paisa "non" extraction ###
##############################





print(f"Uploading PAISA corpus...")
# upload the Italian corpus
'''with open(r"../data/paisa.raw.utf8", encoding='utf8') as infile:
    paisa = infile.read()

print(f"Extracting Wikipedia texts from PAISA...")
# from the corpus, select all texts containing "wiki" in their tag's url
wiki_pattern = r"<text.*wiki.*(?:\n.*)+?\n</text>\n" 
paisa_wiki = re.findall(wiki_pattern, paisa)
dump(paisa_wiki, "../Inputs/paisa_wiki.joblib")
'''
#print(f"Number of texts from a site containing 'wiki' in their URL: {len(paisa_wiki)}")
#paisa_wiki = paisa

paisa_wiki = load("../Inputs/paisa_wiki.joblib")
print(f"Extracting sentences from paisa_wiki...")
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
print(f"Frasi trovate : {len(sent)}")
#print(f"Number of sentences: {len(sent)}")


print(f"Extracting negative sentences from paisa_wiki...")
# splitting the sentences above into two lists:
sent_pos = []
sent_neg = []

# pattern to find the negation in a sentence
neg_patt = r"\b[Nn]on\b"  

for s in sent:
  matches = re.search(neg_patt, s)
  if matches:
    sent_neg.append(s)
  else:
    sent_pos.append(s)

print(f"Frasi positive : {len(sent_pos)} - Frasi negative : {len(sent_neg)}")


size_test = min(size_test, len(sent_neg), len(sent_pos))
print(f"Size test = {size_test}")

shuffle(sent_neg)
shuffle(sent_pos)



# select a fixed numb of sentences to test
sent_neg = sent_neg[:size_test]
sent_pos = sent_pos[:size_test]

dump(sent_neg, "../Inputs/paisa_sent_neg.joblib")
dump(sent_pos, "../Inputs/paisa_sent_pos.joblib")


print(f"Extracting CLS encodings for PAISA sentences...")
### extract CLS
# for each set of sentences, we encode each sentence

cls_encodings_neg = []
cls_encodings_pos = []
all_cls_encodings = []

for sent_list in [sent_neg, sent_pos]:
  m = 0
  all_cls_encodings = []
  for sentence in sent_list:
    sentence_encoded = tokenizer.encode_plus(sentence, padding=True, add_special_tokens=True, return_tensors="pt").to(device)
    if m == 0:
      print(sentence)

    # then extract only the outputs for each sentence
    with torch.no_grad():
      tokens_outputs = model(**sentence_encoded)


    # for each set of outputs we only keep the one of the CLS token, namely the first token of each sentence
    embeddings = tokens_outputs[0]
    cls_encodings = embeddings[:, 0, :]

    m+=1
    cls_encodings = cls_encodings.cpu().numpy()
    if m == 1:
        all_cls_encodings = cls_encodings
        print(cls_encodings.shape)
    if m > 1:
        all_cls_encodings = np.vstack((all_cls_encodings,cls_encodings))
    if m % 200 == 0:
       print(f"Encoded: {m}")
   

  if sent_list == sent_neg:
    cls_encodings_neg = all_cls_encodings
  elif sent_list == sent_pos:
    cls_encodings_pos = all_cls_encodings



#train = torch.zeros(cls_encodings_neg.shape[0]*2, cls_encodings_neg.shape[1])
#train[cls_encodings_neg.shape[0]] = cls_encodings_neg[:9000]
#train = train.append(cls_encodings_pos[:9000])

# we use 90% of data as training and 10% as test
train_size = round(size_test*0.9)
train = np.concatenate((cls_encodings_pos[:train_size], cls_encodings_neg[:train_size]), 0) # shape num_sent x 768
print(f"Train shape : {train.shape}")
labels = np.concatenate((np.zeros(train_size), np.ones(train_size)))
test = np.concatenate((cls_encodings_pos[train_size:], cls_encodings_neg[train_size:]), 0)
print(f"Test shape : {test.shape}")
test_size = int(size_test - train_size)
test_lab = np.concatenate((np.zeros(test_size), np.ones(test_size)))


# data normalization
scaler = StandardScaler()
scaler.fit(train)
dati_scaled = scaler.transform(train)
dump(scaler, f"../Inputs/scaler.joblib")

X = dati_scaled 
test = scaler.transform(test)
#print(test)
#print(test_lab)

y = labels
    
X, y = skshuffle(X, y, random_state=42)





########################################
### classifier creation and training ###
########################################


paisa_result =[]

print(f"Training and testing MLP...")
# set up the MLP classifier
# solver : adam or sgd
# hidden_layer_sizes : 40,40 or 350,350
# alpha : between 1e-5 and 1e-3
n=0
for hl in [(350,350),(40,40)]:
  for a in [1e-5, 1e-4, 1e-3]:
    for solv in ["adam", "sgd"]:
      n+=1
      clf = MLPClassifier(solver = solv, alpha = a,
                    hidden_layer_sizes=hl, random_state = 1)

      # train on data
      clf = clf.fit(X, y)

      # see predictions on the dataset
      predicted = clf.predict(test)
      right_pred = clf.score(test, test_lab)
      tn, fp, fn, tp = confusion_matrix(test_lab, predicted).ravel()
      paisa_result.append(f"Method\t{solv}\nNb hidden layers\t{str(hl)}\nAlpha\t{str(a)}\nScore\t{right_pred}\nTrue neg\t{tn}\nFalse pos\t{fp}\nFalse neg\t{fn}\nTrue pos\t{tp}\n\n")

     
      dump(clf, f"../Inputs/non_classifier_{n}.joblib")
    

print("PAISA' TEST\n")
for scores in paisa_result:
   print(scores)