import json


def adapt_text(example,bert_tokenizer,trainWithMasked):
    tokens = example["tokens"]
    segment_ids = example["segment_ids"]
    if "1" in segment_ids:
        # 1 in segment ids
        return 3
    masked_lm_positions = example["masked_lm_positions"]
    masked_lm_labels = example["masked_lm_labels"]

    if tokens[-2] not in [".", "!", '"', '?', ";", ")", ":"]:
        # "no end of sentence : {tokens[-2]}"
        return 33


    assert tokens[-1] == "[SEP]"
    if tokens.count("[MASK]") > 1:
        return 2

    if len(masked_lm_labels) > 1:
        # print(f"too much maskes in {tokens}")
        return 5

    if len(bert_tokenizer.tokenize(masked_lm_labels[0])) > 1:
        assert bert_tokenizer.convert_tokens_to_ids(masked_lm_labels[0]) == 100
        # print(f"too much toks to predict in {tokens}")
        return 1

    if tokens.count(masked_lm_labels[0]) > 1:
        # print(f"too much occurrences in {tokens}")
        return 7

    if masked_lm_labels[0] == "%":
        return 99

    sent_1 = []
    sent_2 = []

    is_1 = True
    for ll in range(len(tokens) - 2):
        element = tokens[1 + ll]
        if is_1:
            sent_1.append(element)
        else:
            sent_2.append(element)
        if element in [".", "!", "?", ";", ":", ")", '"']:
            if not element == tokens[-2]:
                # is_1 = True
                # print("end of sentence")
                ccc = 0

            elif ll < len(tokens) - 3:
                if tokens[2 + ll] == tokens[1]:
                    is_1 = False
                elif tokens[2 + ll] == "do" and tokens[3 + ll] == "not" and tokens[4 + ll] == tokens[1]:
                    return 6

                elif tokens[2 + ll] == "[MASK]":
                    if tokens[3 + ll] == tokens[2]:
                        is_1 = False
                    if tokens[3 + ll] == "cannot":
                        is_1 = False

                    elif tokens[3 + ll] in ["did", "do", "does"] and tokens[4 + ll] == "not":
                        is_1 = False

    if len(sent_1) == 0 or len(sent_2) == 0:
        return 44

    a = sent_1.count("not")
    b = sent_2.count("not")

    if b < a:
        return 8

    sent_2_m = sent_2.copy()

    for i, token in enumerate(sent_2):
        if token == "[MASK]":
            sent_2[i] = masked_lm_labels[0]
            break

    if trainWithMasked:
        sent_1_m = sent_1.copy()
        for i, token in enumerate(sent_1_m):
            if token == masked_lm_labels[0]:
                sent_1_m[i] = "[MASK]"
                break
        new_example = {"sent_1": sent_1_m, "sent_2": sent_2_m, "masked_lm_labels": masked_lm_labels, "maskToken" : bert_tokenizer.mask_token}

    else:
        new_example = {"sent_1": sent_1, "sent_2": sent_2, "masked_lm_labels": masked_lm_labels}

    '''print("\n\n####\n\n")

    print(sent_1)
    print("\n")
    print(sent_2)'''


    return new_example


def get_data(path_data, bert_tokenizer, trainWithMasked):
    list_data = []
    #with path_data.open() as f:
    with open(path_data, "r") as f:
        for line in f:
            example = json.loads(line)
            dd = adapt_text(example, bert_tokenizer, trainWithMasked)
            if type(dd) != int:
                maskedTok = dd["masked_lm_labels"]
                sizeMaskedTok = len(bert_tokenizer.tokenize(maskedTok[0]))
                if not(maskedTok[0] in ["something", "somewhere"] or  len(maskedTok) != 1 or sizeMaskedTok > 1):
                    if (dd["sent_2"].count(maskedTok[0]) == 1 or dd["sent_1"].count(maskedTok[0]) == 1) or (dd["sent_2"].count(maskedTok[0]) == 0 or dd["sent_1"].count(maskedTok[0]) == 0 and trainWithMasked) :
                        list_data.append(dd)
    return list_data