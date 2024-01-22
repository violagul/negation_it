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





path = r"../Inputs"
list_verbs = load(f"{path}/base_verbs.joblib")

list_conj_vb = []

for verb in list_verbs:
   conj_vb = get_conj(verb)
   list_conj_vb.append(conj_vb)




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
pattern = r"[.?!] [A-Z][a-z ]*[,:]?[a-z ]+[,:]?[a-z ][,:]?[a-z]+\. \b"  # finds kind of acceptable sentences


for text in paisa_wiki[:100]:
  found = re.findall(pattern, text)
  for elem in  found:
    if len(elem) > 40:
      for verb in list_conj_vb:
         if verb in elem:
            sent.append(elem[:1])
            if len(sent)%1000 ==0:
               print(elem[:1])
               print(f"{len(sent)} frasi trovate")
  if len(sent)> size_test*100:
    break

print(sent[0], sent[100], sent[1000])
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
    if len(sent_neg)%500 ==0:
       print(f"{len(sent_neg)} frasi neg")
  else:
    sent_pos.append(s)

print(f"Frasi positive : {len(sent_pos)} - Frasi negative : {len(sent_neg)}")


size_test = min(size_test, len(sent_neg), len(sent_pos))
print(f"Size test = {size_test}")

shuffle(sent_neg)
shuffle(sent_pos)

print(sent_neg[0], sent_neg[100],sent_neg[1000],)

# select a fixed numb of sentences to test
sent_neg = sent_neg[:size_test]
sent_pos = sent_pos[:size_test]

dump(sent_neg, "../Inputs/paisa_sent_neg2.joblib")
dump(sent_pos, "../Inputs/paisa_sent_pos2.joblib")