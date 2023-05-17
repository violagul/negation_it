import torch
from .chech_conjug import check_conjugation

def mask_prediction(sentence_available, tokenizer, model, device, conj_act_token_id):
    encoded_sentence = tokenizer.encode(sentence_available, return_tensors="pt").to(device)
    mask_token_index = torch.where(encoded_sentence == tokenizer.mask_token_id)[1]
    token_logits = model(encoded_sentence)[0]
    mask_token_logits = token_logits[0, mask_token_index, :]
    indice_act, sorted, probas_tok = get_indice_act(mask_token_logits, conj_act_token_id)
    top_token = torch.topk(mask_token_logits, 1, dim=1).indices[0].tolist()
    predicted_token=tokenizer.decode([top_token[0]])
    return predicted_token, mask_token_logits[0], indice_act, probas_tok




def get_indice_act(token_logits, conj_act_token_id):

    probas = torch.softmax(token_logits[0], dim=0)

    sorted, indices = torch.sort(probas, descending=True)

    indx = (indices == conj_act_token_id).nonzero()

    return indx.item(), indices, sorted[indx]





def batch_mask_prediction(batch_sentences, tokenizer, model, device, size_batches = 8):

    all_predictions = []

    nb_batches = len(batch_sentences) // size_batches
    for k in range(nb_batches):
        current_batch = batch_sentences[k*size_batches:(k+1)*size_batches]
        predicted_tokens = encode_batch(current_batch, tokenizer, model, device)
        all_predictions.extend(predicted_tokens)



    if len(batch_sentences) % size_batches != 0:
        last_batch = batch_sentences[nb_batches*size_batches:]
        predicted_tokens = encode_batch(last_batch, tokenizer, model, device)
        all_predictions.extend(predicted_tokens)


    return all_predictions




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
                         "verb": good_verb, "current_pronouns_maj": current_pronouns_maj}

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
        mask_tokens_logits = tokens_logits['logits'][ mask_tokens_index]
        #print(mask_tokens_logits.shape)

        # get the k highest logits
        top_tokens = torch.topk(mask_tokens_logits, 1, dim=1).indices#.tolist()        
        #print(top_tokens)

        # decode the batch of tokens, i.e. get the predicted tokens (each token is represented by an index in the vocabulary)
        predicted_tokens = tokenizer.batch_decode(top_tokens) 
        #print(predicted_tokens)

    return predicted_tokens