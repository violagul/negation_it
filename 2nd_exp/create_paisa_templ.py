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





print("Loading paisa_wiki sentences...")
'''paisa_sent_neg = load("../Inputs/paisa_sent_neg.joblib")
paisa_sent_pos = load("../Inputs/paisa_sent_pos.joblib")'''
paisa_sent_neg = load("../Inputs/paisa_sent_neg2.joblib")
paisa_sent_pos = load("../Inputs/paisa_sent_pos2.joblib")




states_list = "Sudafrica	Algeria 	Angola	Benin	Botswana	Burkina Faso	Burundi	Camerun	Capo Verde	Repubblica Centrafricana	Comore	Costa d’Avorio	Repubblica del Congo	Repubblica Democratica del Congo	Gibuti	Egitto	Eritrea	Etiopia	Gabon	Gambia	Ghana	Guinea	Guinea-Bissau	Guinea Equatoriale	Kenia	Lesotho	Liberia	Libia	Madagascar	Malawi	Mali	Marocco	Mauritania	Mauritius	Mozambico	Namibia	Niger	Nigeria	Uganda	Ruanda	São Tomé e Príncipe	Senegal	Seychelles	Sierra Leone	Somalia	Sudan	Sudan del Sud	Swaziland	Tanzania	Ciad	Togo	Tunisia	Zambia	Zimbabwe	Antigua e Barbuda	Argentina	Bahamas	Barbados	Belize	Bermuda	Bolivia	Brasile	Canada	Cile	Colombia	Costa Rica	Cuba	Repubblica dominicana	Dominica	Ecuador	Stati Uniti d’America	Grenada	Groenlandia	Guatemala	Guyana	Haiti	Honduras	Giamaica	Messico	Nicaragua	Panamá	Paraguai	Perù	Porto Rico	Saint Kitts e Nevis	Saint Vincent e Grenadine	Santa Lucia	El Salvador	Suriname	Trinidad e Tobago	Uruguay	Venezuela	Afghanistan	Arabia Saudita	Armenia	Azerbaigian	Bahrain	Bangladesh	Bhutan	Birmania	Brunei	Cambogia	Cina	Corea del Nord	Corea del Sud	Emirati Arabi Uniti	Georgia	Giappone	Giordania	India	Indonesia	Iran	Iraq	Israele	Kazakistan	Kirghizistan	Kuwait	Laos	Libano	Malesia	Maldive	Mongolia	Nepal	Oman	Uzbekistan	Pakistan	Palestina	Filippine	Qatar	Russia	Singapore	Siria	Sri Lanka	Tagikistan	Taiwan	Thailandia	Timor Est	Turchia	Turkmenistan	Vietnam	Yemen	Albania	Germania	Andorra	Inghilterra	Austria	Belgio	Bielorussia	Bosnia ed Erzegovina	Bulgaria	Cipro	Croazia	Danimarca	Scozia	Spagna	Estonia	Isole FærØer	Finlandia	Francia	Gibilterra	Grecia	Ungheria	Irlanda	Islanda	Italia	Kosovo	Lettonia	Liechtenstein	Lituania	Lussemburgo	Macedonia	Malta	Isola di Man	Moldavia	Monaco	Montenegro	Norvegia	Paesi Bassi	Galles	Polonia	Portogallo	Regno Unito	Romania	San Marino	Serbia	Slovacchia	Slovenia	Svezia	Svizzera	Repubblica Ceca	Ucraina	Vaticano	Australia	Figi	Kiribati	Isole Marshall	Micronesia	Nauru	Nuova Zelanda	Palau	Papua Nuova Guinea	Isole Salomone	Samoa	Tonga	Tuvalu	Vanuatu"
states_list = states_list.split("\t")


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

fprofarray = build_array(fProf_file)
mprofarray = build_array(mProf_file)
fnamearray = build_array(fName_file)
mnamearray = build_array(mName_file)



negative_ex = []
positive_ex = []
m=0
for sent_list in [paisa_sent_neg, paisa_sent_pos]:
    n=0
    if sent_list == paisa_sent_neg:
        print("Frasi negative...")
    if sent_list == paisa_sent_pos:
        print("Frasi positive...")

    state_sent = []
    fname_sent = []
    mname_sent = []
    fprof_sent = []
    mprof_sent = []
    fname_state_sent = []
    mname_state_sent = []
    fname_prof_sent = []
    mname_prof_sent = []
    for sent in sent_list:
        n+=1
        if len(sent) < 55:
            m+=1
        if n%750==0:
            print(f"sent n. {n}")

        for state in states_list:
            if state in sent:
                if len(state_sent) < 50:
                    state_sent.append(sent)
        
        for namelist in [[fnamearray, fname_sent], [mnamearray, mname_sent]]:
            for name in namelist[0]:
                if name in sent:
                    if len(namelist[1])<300:
                        namelist[1].append(sent)
        
        for proflist in [[fprofarray, fprof_sent],[mprofarray, mprof_sent]]:
            for prof in proflist[0]:
                if prof in sent:
                    if len(proflist[1]) < 50:
                        proflist[1].append(sent)
        
        for sent_w_fname in fname_sent:
            for state in states_list:
                if len(fname_state_sent) < 20:
                    if state in sent_w_fname:
                        fname_state_sent.append(sent_w_fname)
            for prof in fprofarray:
                if len(fname_prof_sent) < 20:
                    if prof in sent_w_fname:
                        fname_prof_sent.append(sent_w_fname)
        
        
        for sent_w_mname in mname_sent:
            for state in states_list:
                if len(mname_state_sent) < 20:
                    if state in sent_w_mname:
                        mname_state_sent.append(sent_w_mname)
            for prof in mprofarray:
                if len(mname_prof_sent) < 20:
                    if prof in sent_w_mname:
                        mname_prof_sent.append(sent_w_mname)

    if sent_list == paisa_sent_neg:
        nstate_sent = state_sent
        nfname_sent = fname_sent
        nmname_sent = mname_sent
        nfprof_sent = fprof_sent
        nmprof_sent = mprof_sent
        nfname_prof_sent = fname_prof_sent
        nmname_prof_sent = mname_prof_sent
        nfname_state_sent = fname_state_sent
        nmname_state_sent = mname_state_sent
    if sent_list == paisa_sent_pos:
        pstate_sent = state_sent
        pfname_sent = fname_sent
        pmname_sent = mname_sent
        pfprof_sent = fprof_sent
        pmprof_sent = mprof_sent
        pfname_prof_sent = fname_prof_sent
        pmname_prof_sent = mname_prof_sent
        pfname_state_sent = fname_state_sent
        pmname_state_sent = mname_state_sent


shuffle(nstate_sent)
shuffle(nfname_sent)
shuffle(nmname_sent)
shuffle(nfprof_sent)
shuffle(nmprof_sent)
shuffle(nmname_prof_sent)
shuffle(nfname_prof_sent)
shuffle(nfname_state_sent)
shuffle(nmname_state_sent)

shuffle(pstate_sent)
shuffle(pfname_sent)
shuffle(pmname_sent)
shuffle(pfprof_sent)
shuffle(pmprof_sent)
shuffle(pfname_state_sent)
shuffle(pmname_state_sent)
shuffle(pfname_prof_sent)
shuffle(pmname_prof_sent)




print("Negative:\n\n")
print(f"State: {len(nstate_sent)}\n{nstate_sent[:3]}\n")
print(f"f names: {len(nfname_sent)}\n{nfname_sent[:3]} \n")
print(f"m names: {len(nmname_sent)}\n{nmname_sent[:3]} \n")
print(f"f profs: {len(nfprof_sent)}\n{nfprof_sent[:3]} \n")
print(f"m profs: {len(nmprof_sent)}\n{nmprof_sent[:3]} \n")
print(f"f name + prof: {len(nfname_prof_sent)}\n{nfname_prof_sent[:3]} \n")
print(f"m name + prof: {len(nmname_prof_sent)}\n{nmname_prof_sent[:3]} \n")
print(f"f name + state: {len(nfname_state_sent)}\n{nfname_state_sent[:3]} \n")
print(f"m name + state: {len(nmname_state_sent)}\n{nmname_state_sent[:3]} \n")
print("\nPositive:\n\n")
print(f"State: {len(pstate_sent)}\n{pstate_sent[:3]} \n")
print(f"f names: {len(pfname_sent)}\n{pfname_sent[:3]} \n")
print(f"m names: {len(pmname_sent)}\n{pmname_sent[:3]} \n")
print(f"f profs: {len(pfprof_sent)}\n{pfprof_sent[:3]} \n")
print(f"m profs: {len(pmprof_sent)}\n{pmprof_sent[:3]} \n")
print(f"f name + prof: {len(pfname_prof_sent)}\n{pfname_prof_sent[:3]} \n")
print(f"m name + prof: {len(pmname_prof_sent)}\n{pmname_prof_sent[:3]} \n")
print(f"f name + state: {len(pfname_state_sent)}\n{pfname_state_sent[:3]} \n")
print(f"m name + state: {len(pmname_state_sent)}\n{pmname_state_sent[:3]} \n")




dump(nstate_sent, "../Inputs/nstate.joblib")
dump(nfname_sent, "../Inputs/nfname.joblib")
dump(nmname_sent, "../Inputs/nmname.joblib")
dump(nfprof_sent, "../Inputs/nfprof.joblib")
dump(nmprof_sent, "../Inputs/nmprof.joblib")
dump(nmname_prof_sent, "../Inputs/nmnameprof.joblib")
dump(nfname_prof_sent, "../Inputs/nfnameprof.joblib")
dump(nfname_state_sent, "../Inputs/nfnamestate.joblib")
dump(nmname_state_sent, "../Inputs/nmnamestate.joblib")

dump(pstate_sent, "../Inputs/pstate.joblib")
dump(pfname_sent, "../Inputs/pfname.joblib")
dump(pmname_sent, "../Inputs/pmname.joblib")
dump(pfprof_sent, "../Inputs/pfprof.joblib")
dump(pmprof_sent, "../Inputs/pmprof.joblib")
dump(pfname_state_sent, "../Inputs/pfnamestate.joblib")
dump(pmname_state_sent, "../Inputs/pmnamestate.joblib")
dump(pfname_prof_sent, "../Inputs/pfnameprof.joblib")
dump(pmname_prof_sent, "../Inputs/pmnameprof.joblib")