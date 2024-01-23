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


print("Downloading sentences...")

#liste vuote commentate
nstate_sent = load(, "../Inputs/nstate.joblib")
nfname_sent = load(, "../Inputs/nfname.joblib")
nmname_sent = load(, "../Inputs/nmname.joblib")
nfprof_sent = load(, "../Inputs/nfprof.joblib")
nmprof_sent = load(, "../Inputs/nmprof.joblib")
#nmname_prof_sent = load(, "../Inputs/nmnameprof.joblib")
#nfname_prof_sent = load(, "../Inputs/nfnameprof.joblib")
#nfname_state_sent = load(, "../Inputs/nfnamestate.joblib")
#nmname_state_sent = load(, "../Inputs/nmnamestate.joblib")

#pstate_sent = load(, "../Inputs/pstate.joblib")
pfname_sent = load(, "../Inputs/pfname.joblib")
pmname_sent = load(, "../Inputs/pmname.joblib")
pfprof_sent = load(, "../Inputs/pfprof.joblib")
pmprof_sent = load(, "../Inputs/pmprof.joblib")
#pfname_state_sent = load(, "../Inputs/pfnamestate.joblib")
#pmname_state_sent = load(, "../Inputs/pmnamestate.joblib")
#pfname_prof_sent = load(, "../Inputs/pfnameprof.joblib")
#pmname_prof_sent = load(, "../Inputs/pmnameprof.joblib")