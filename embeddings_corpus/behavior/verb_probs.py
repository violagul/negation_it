import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import json
import torch
from get_data import get_data, adapt_text
from transformers import BertTokenizer, AutoModelForMaskedLM
from extract_top_pred import extract_top_tokens, detokenize, extract_logits
from joblib import dump, load
import torch.nn.functional as F
import torch.nn as nn
kl_loss = nn.KLDivLoss(reduction="batchmean")

device = torch.device("cuda") if torch.cuda.is_available() else torch.devide("cpu")

'''preds = load(r"predict_lists.joblib")
print(preds)'''

kl_scores = load(r"kl_scores.joblib")
print(len(kl_scores))
print(len(kl_scores["be"]))
print(str(kl_scores)[:300])


data = load(r"/data/dkletz/data/final_data.joblib")
#print(data)

path = r"/data/dkletz/data/wiki20k_negated_withref_UL_pregenerated/epoch_0.json" # usare verbdata.joblib o qlcs di simile
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
model = AutoModelForMaskedLM.from_pretrained("bert-base-cased").to(device)

sent_data = get_data(path, tokenizer, False)

'''print(f"Type data : {type(sent_data)}")
print(f"List of : {type(sent_data[0])}")
print(f"List contains : {len(sent_data)} dictionaries")
print(f"Each dictionary contains {len(list(sent_data[0].keys()))} sentences ({list(sent_data[0].keys())})")'''


### sent_data:
# list of 14635 dict
# each dict contains 3 keys: 'sent_1', 'sent_2', 'masked_lm_labels'
# sent_1 = affirmative sentence
# sent_2 = negative sentence
# masked_lm_labels = the token of the synt object of the sentence


          ######## MASKED #########
### create new list with masked sentences:
# for each sent:
# target the obj
# change it into MASK
# extract the top k predictions for that token
# compare the affirm and neg top predictions

sent_data_masked = []
mask_token = "[MASK]"
for sent_set in data:
    sent_set_masked = {}
    to_mask = sent_set["negated_verb"]
    ok = True
    for polarity in ["sent_1", "sent_2"]:
        if to_mask in sent_set[polarity]:
            sent_set_masked[polarity] = [mask_token if x == to_mask else x for x in sent_set[polarity]]
        else:
            if sent_set["inifintive"] in sent_set[polarity]:
                sent_set_masked[polarity] = [mask_token if x == sent_set["inifintive"] else x for x in sent_set[polarity]]
            else:
                ok = False
                continue

    if ok:
        sent_set_masked["negated_verb"] = to_mask
        sent_set_masked["infinitive"] = sent_set["inifintive"]
        sent_data_masked.append(sent_set_masked)

print("masked")

predictions = []
n=0
'''print("Predicting...")
for sent_set in sent_data_masked:
    n+=1 ## counter
    pred_dict = {}
    ## affirmatives
    sent = detokenize(sent_set["sent_1"])
    top_id = extract_top_tokens(sent, 30, tokenizer, model, device)
    top_aff = [tokenizer.convert_ids_to_tokens(id) for id in top_id]
    pred_dict["aff"] = top_aff
    ## negatives
    sent = detokenize(sent_set["sent_2"])
    top_id = extract_top_tokens(sent, 30, tokenizer, model, device)
    top_neg = [tokenizer.convert_ids_to_tokens(id) for id in top_id]
    pred_dict["neg"] = top_neg
    ## add the masked token to the dict
    pred_dict["original"] = sent_set["masked_lm_labels"]
    predictions.append(pred_dict)
    
    if n%200 == 0:
        print(f"{n} of 14635")

dump(predictions, f"predict_lists.joblib")
print(predictions[0])'''



kl_dict = {}
for sent_set in sent_data_masked:
    n+=1 ## counter
    ## affirmatives
    sent = detokenize(sent_set["sent_1"])
    with torch.no_grad():
        logits_pos = extract_logits(sent, tokenizer, model, device)
        '''top_aff = [tokenizer.convert_ids_to_tokens(id) for id in top_id]
        pred_dict["aff"] = top_aff'''
        ## negatives
        sent = detokenize(sent_set["sent_2"])
        logits_neg = extract_logits(sent, tokenizer, model, device)
        '''top_neg = [tokenizer.convert_ids_to_tokens(id) for id in top_id]
        pred_dict["neg"] = top_neg'''
        ## add the masked token to the dict
        '''pred_dict["original"] = sent_set["masked_lm_labels"]
        predictions.append(pred_dict)'''
        
        if n%200 == 0:
            print(f"{n} of 14635")



    inputs = F.log_softmax(logits_neg, dim=1)
    targets = F.softmax(logits_pos, dim=1)
    output = kl_loss(inputs, targets)
    '''print(logits_neg.shape)
    print(logits_pos.shape)
    print(inputs.shape)
    print(targets.shape)'''
    #print(output)
    #print(sent_set)
    if "infinitive" in sent_set.keys():
        verb = sent_set["infinitive"]
        if verb not in kl_dict:
            kl_dict[verb] = []
        kl_dict[verb].append(output)

print(len(kl_dict))


dump(kl_dict, r"kl_scores.joblib")

### classi di perc neg -> x ogni classe fare una lista dei kl dei verbi -> fare la distribuzione

'''
# input should be a distribution in the log space -> neg distrib (logits)
input = F.log_softmax(torch.randn(3, 5, requires_grad=True), dim=1)
# Sample a batch of distributions. Usually this would come from the dataset -> target is positive distribution
target = F.softmax(torch.rand(3, 5), dim=1)
output = kl_loss(input, target)

poi faremo una distribuzione di sta roba risp al ranking del vb
'''