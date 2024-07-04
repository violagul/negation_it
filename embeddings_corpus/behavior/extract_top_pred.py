import torch

def extract_top_tokens(sentence_to_encode, n, tokenizer, model, device):

    """ This function takes in a sentence with a mask-token
    (as well as a number of other things), lets the model passed
    predict the token and returns an array with the top n predicted token.
    """

    encoded_sentence = tokenizer.encode(sentence_to_encode, return_tensors="pt").to(device)

    mask_token_index = torch.where(encoded_sentence == tokenizer.mask_token_id)[1]

    token_logits = model(encoded_sentence)[0]

    mask_token_logits = token_logits[0, mask_token_index, :]

    top_tokens = torch.topk(mask_token_logits, n, dim=1).indices[0].tolist()

    predicted_token = tokenizer.decode([top_tokens[0]])


    return top_tokens



def extract_logits(sentence_to_encode, tokenizer, model, device):

    """ This function takes in a sentence with a mask-token
    (as well as a number of other things), lets the model passed
    predict the token and returns an array with the top n predicted token.
    """

    encoded_sentence = tokenizer.encode(sentence_to_encode, return_tensors="pt").to(device)
    #print(sentence_to_encode)

    mask_token_index = torch.where(encoded_sentence == tokenizer.mask_token_id)[1]
    #print(mask_token_index)

    token_logits = model(encoded_sentence)[0]
    #print(token_logits.shape)

    mask_token_logits = token_logits[0, mask_token_index, :]
    #print(mask_token_logits)

    return mask_token_logits


def detokenize(list_of_tokens):
    sent = ""
    for tok in list_of_tokens:
        if tok.startswith("##"):
            sent += tok[2:]
        else:
            sent += " " + tok
    sent = sent[1:]
    return sent



def extract_token_rank(sentence_to_encode, tokenizer, model, device, token_to_find):

    """ This function takes in a sentence with a mask-token
    (as well as a number of other things), lets the model passed
    predict the token and returns an array with the top n predicted token.
    """

    encoded_sentence = tokenizer.encode(sentence_to_encode, return_tensors="pt").to(device)

    mask_token_index = torch.where(encoded_sentence == tokenizer.mask_token_id)[1]

    token_logits = model(encoded_sentence)[0]

    mask_token_logits = token_logits[0, mask_token_index, :]
    

    sorted_logits = torch.argsort(mask_token_logits, descending = True, dim = -1)[0]

    tok_id = tokenizer.convert_tokens_to_ids(token_to_find)
    #print(f"tok_id {tok_id}")

    token_rk = (sorted_logits == tok_id).nonzero()[0][0].item()
    #print(f"token_rk {token_rk}")


    return token_rk
