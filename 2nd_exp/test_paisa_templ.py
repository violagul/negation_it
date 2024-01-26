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


# frasi positive rese negative da me
pfname = ["FNOM non viene arrestata dai carabinieri sotto gli occhi delle amiche.", 
          "FNOM comunque non lo ritiene una brava persona su cui poter fare affidamento.", 
          "FNOM non riesce a sorprendere il pubblico e la giuria con la sua voce potente ed emozionante.", 
          "FNOM non rimane incinta e decide di abortire.", 
          "FNOM organizza alcuni espedienti e alla fine riesce a non farli riavvicinare.", 
          "FNOM rifiuta di ascoltarlo ancora e non se ne va.", 
          "FNOM ha degli strani malesseri e non pensa di essere incinta.", 
          "FNOM non torna a casa e si prepara per la sua festa di compleanno."]
pmname = ["MNOM non corre velocemente a vedere che succede.", 
          "MNOM stesso non venne poco dopo assassinato.",
          "MNOM non fece esiliare tutta la famiglia.",
          "MNOM non viene arrestato e inizia un lungo processo.",
          "MNOM non riprende gli studi raccogliendo successi e premi.",
          "MNOM che non amava molto la figlia, decise di soddisfare le sue richieste.",
          "MNOM non mostra una iniziale inclinazione per la filosofia.",
          "MNOM decise di non sposare la sua ex amante.",
          "MNOM non la comprende, e decide di lasciar stare.",
          "MNOM non sarebbe stato assassinato dopo una messa."]
pfprof = ["Ha un fratello e un giorno non spera di diventare una FPROF di successo."]
pmprof = ["Accompagnato da un MPROF di fiducia non intraprese a piedi la sua escursione.", 
          "Esitante a definirsi un MPROF, non preferisce pensarsi come un cantante.", 
          "Barker, non fu arrestato per aver cercato di vendere un quarto di oncia di marijuana a un MPROF sotto copertura.", 
          "Le sue pennellate sottili e incisive non hanno i caratteri inquieti di un MPROF alla ricerca di nuove soluzioni spaziali e formali." 
          "Leonhard infatti non era probabilmente solo un MPROF.", 
          "Altre leggende non fanno di lui un MPROF prima di diventare soldato."]

# frasi negative rese positive da me
nfname = ["FNOM porta avanti la sua seconda vita, ma si accorge che qualcuno la sta seguendo.", 
          "FNOM cerca di stare vicina al marito, pur amandolo.", 
          "FNOM crede che possa essere lui il ragazzo del suo primo vero bacio, che ha mai dato.", 
          "FNOM ha lasciato alcun ritratto similare di qualsivoglia altro principe crociato.", 
          "FNOM intano riesce a controllarsi e scarta tutti i regali di nozze.", 
          "FNOM ha paura: affronta le apparizioni e le fa sparire con la sua determinata reazione.", 
          "FNOM riabilita il suo nome ma prima di aver scoperto che essite una terza macchina che produce carte di credito false."]
nmname = ["MNOM vorrebbe parlare ma la ragazza ne ha voglia e ha fretta, e si fa riaccompagnare al suo posto sulla strada.", 
          "MNOM sostiene che gli importa egli vuole vivere bene e si fa domande su chi paghi.", 
          "MNOM ebbe cura di tornare in patria in questo periodo, sapendo di essere indesiderato.", 
          "MNOM si sarebbe mostrato reticente, pur lesinandogli denaro e rifornimenti.", 
          "MNOM comprende il motivo che ha spinto la giovane a chiedergli una cosa simile.", 
          "MNOM resta a vivere nella casa dove avevano abitato tutti assieme, rassegnandosi alla perdita della moglie.", 
          "MNOM risponde di sapere chi sia e il giovane gli racconta la sua storia.", 
          "MNOM fu fatto prigioniero e giustiziato: apparentemente le sue truppe opposero molta resistenza.", 
          "MNOM inoltre parla nel sonno nella versione giapponese.", 
          "MNOM invece, si era mosso celermente con il suo esercito impacciato e armato alla leggera."]
nfprof = ["Egli sopporta questa situazione e considera la madre una FPROF mediocre.", 
          "Perfare un esempio una FPROF poteva comperare direttamente il tessuto, che era venduto esclusivamente dal fabbricante.", 
          "Le sue regole coincidono con quel mondo che si era immaginato ed erra alla ricerca di qualcosa che pensa di aver trovato in una FPROF."]
nmprof = ["Sarebbe probabilmente divenuto un MPROF se la filosofia fosse stata la sua innata passione.", 
          "Conoscendo il MPROF, i due pensarono che fosse un cantante e che fosse lui a cantare il brano.", 
          "Secondo il loro punto di vista un MPROF poteva rimanere legato a concetti obsoleti, ma doveva dare libero sfogo al proprio estro.", 
          "Fulci voleva un MPROF noto per quella parte.", 
          "Il primo era un MPROF e la seconda era americana: andavano quindi doppiati.", 
          "Addirittura vennero mostrate in video le loro buste paga, che avrebbero dovuto confermare il fatto che questi guadagnassero gli stipendi di un MPROF."]



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



for lista in [pfname, nfname]:
    templates = {}
    for frase in lista:
        exfrase = []
        for nome in fnamearray:
            fnames = re.sub("FNOM", nome, frase)
            exfrase.append(fnames)
        templates[frase] = exfrase
    if lista == pfname:
        nfnametempl = templates
        # n perché il template è negativo, anche se derivato da frasi positive
    if lista == nfname:
        pfnametempl = templates
        # n perché il template è positivo, anche se derivato da frasi negative


for lista in [pmname, nmname]:
    templates = {}
    for frase in lista:
        exfrase = []
        for nome in mnamearray:
            mnames = re.sub("MNOM", nome, frase)
            exfrase.append(mnames)
        templates[frase] = exfrase
        if lista == pmname:
            nmnametempl = templates
            # n perché il template è negativo, anche se derivato da frasi positive
        if lista == nmname:
            pmnametempl = templates
            # n perché il template è positivo, anche se derivato da frasi negative


for lista in [pfprof, nfprof]:
    templates = {}
    for frase in lista:
        exfrase = []
        for profs in fprofarray:
            fprofs = re.sub("FPROF", profs, frase)
            exfrase.append(fprofs)
        templates[frase] = exfrase
    if lista == pfprof:
        pfproftempl = templates
        # n perché il template è negativo, anche se derivato da frasi positive
    if lista == nfprof:
        nfproftempl = templates
        # n perché il template è positivo, anche se derivato da frasi negative


for lista in [pmprof, nmprof]:
    templates = {}
    for frase in lista:
        exfrase = []
        for profs in mprofarray:
            mprofs = re.sub("MPROF", profs, frase)
            exfrase.append(mprofs)
        templates[frase] = exfrase
    if lista == pmprof:
        pmproftempl = templates
        # n perché il template è negativo, anche se derivato da frasi positive
    if lista == nmprof:
        nmproftempl = templates
        # n perché il template è positivo, anche se derivato da frasi negative




neg_paisa_templates = nmproftempl.copy()
for dictionary in [nfproftempl, nmnametempl, nfnametempl]:
    for key, value in dictionary.items():
        neg_paisa_templates[key] = value

pos_paisa_templates = pmproftempl.copy()
for dictionary in [pfproftempl, pmnametempl, pfnametempl]:
    for key, value in dictionary.items():
        pos_paisa_templates[key] = value
#neg_paisa_templates = nfnametempl + nmnametempl + nfproftempl + nmproftempl
#print(f"Templates negativi\t{len(neg_paisa_templates)}\n{neg_paisa_templates[0]}\n{neg_paisa_templates[-1]}\n\n")
#pos_paisa_templates = pfnametempl + pmnametempl + pfproftempl + pmproftempl
#print(f"Templates positivi\t{len(pos_paisa_templates)}\n{pos_paisa_templates[0]}\n{pos_paisa_templates[-1]}\n\n")
'''
size_test = min(3000, len(neg_paisa_templates))
print(f"Size test neg: {size_test}")
neg_paisa_templates = neg_paisa_templates[:size_test]
neg_paisa_lab = np.ones(size_test)

size_test = min(3000, len(pos_paisa_templates))
print(f"Size test pos: {size_test}")
pos_paisa_templates = pos_paisa_templates[:size_test]
pos_paisa_lab = np.zeros(size_test)
'''




all_cls_encodings = []
for templ_list in [neg_paisa_templates, pos_paisa_templates]:
  m = 0 
  for templ_sent, template_set in templ_list.items():
    encodings_dict = {}
    for sentence in template_set:
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
        if m % 200 == 0:
            print(str(m) + "\textracted")
   
    encodings_dict[templ_sent] = all_cls_encodings
   
   
  if templ_list == pos_paisa_templates:
      pos_paisa_cls = encodings_dict
  elif templ_list == neg_paisa_templates:
      neg_paisa_cls = encodings_dict



neg_paisa_lab = {}
for templ_sent, templ_list in neg_paisa_cls.items():
    np.random.shuffle(templ_list)
    size_test = min(3000, len(templ_list))
    neg_paisa_cls[templ_sent] = templ_list[size_test]
    neg_paisa_lab[templ_sent] = np.ones(size_test)


pos_paisa_lab = {}
for templ_sent, templ_list in pos_paisa_cls.items():
    np.random.shuffle(templ_list)
    size_test = min(3000, len(templ_list))
    pos_paisa_cls[templ_sent] = templ_list[size_test]
    pos_paisa_lab[templ_sent] = np.zeros(size_test)


print(str(pos_paisa_cls)[:120])
print(str(pos_paisa_lab)[:120])
print(str(neg_paisa_cls)[:120])
print(str(neg_paisa_lab)[:120])




'''
#data normalization
scaler = load(f"../Inputs/scaler.joblib")
paisa_pos_templ_test = scaler.transform(pos_paisa_cls)
paisa_neg_templ_test = scaler.transform(neg_paisa_cls)



paisa_neg_res = []
paisa_pos_res = []


print("Classifier working...")   
for n in range(1, 13):
   clf = load(f"../Inputs/non_classifier2_{n}.joblib")
   
   
   predicted = clf.predict(paisa_pos_templ_test)
   right_pred = clf.score(paisa_pos_templ_test, pos_paisa_lab)
   if len(confusion_matrix(pos_paisa_lab, predicted).ravel()) == 4:
       tn, fp, fn, tp = confusion_matrix(pos_paisa_lab, predicted).ravel()
       paisa_pos_res.append(f"Score\t{right_pred}\n\nTrue neg\t{tn}\nFalse pos\t{fp}\nFalse neg\t{fn}\nTrue pos\t{tp}\n\n")
   else:
       conf_matr = confusion_matrix(pos_paisa_lab, predicted).ravel()
       paisa_pos_res.append(f"Score\t{right_pred}\n\nConfusion matrix : {conf_matr}\n\n")
   #template_result.append(f"Method\t{solv}\nNb hidden layers\t{str(hl)}\nAlpha\t{str(a)}\nScores\t{right_pred}\n\nTrue neg\t{tn}\nFalse pos\t{fp}\nFalse neg\t{fn}\nTrue pos\t{tp}\n\n")



   predicted = clf.predict(paisa_neg_templ_test)
   right_pred = clf.score(paisa_neg_templ_test, neg_paisa_lab)
   if len(confusion_matrix(neg_paisa_lab, predicted).ravel()) == 4:
       tn, fp, fn, tp = confusion_matrix(neg_paisa_lab, predicted).ravel()
       paisa_neg_res.append(f"Score\t{right_pred}\n\nTrue neg\t{tn}\nFalse pos\t{fp}\nFalse neg\t{fn}\nTrue pos\t{tp}\n\n")
   else:
       conf_matr = confusion_matrix(neg_paisa_lab, predicted).ravel()
       paisa_neg_res.append(f"Score\t{right_pred}\n\nConfusion matrix : {conf_matr}\n\n")



print("PAISA TEMPLATE TEST NEG\n\n")
for scores in paisa_neg_res:
   print(scores)


print("PAISA TEMPLATE TEST POS\n\n")
for scores in paisa_pos_res:
   print(scores)
'''