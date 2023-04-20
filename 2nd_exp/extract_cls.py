# -*- coding: utf-8 -*-
"""extract_cls.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1a26fT7hBZvNzdbzNuHZ-S0EEr-yVAEfq
"""

#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

model = AutoModelForMaskedLM.from_pretrained('dbmdz/bert-base-italian-cased')
tokenizer = AutoTokenizer.from_pretrained('dbmdz/bert-base-italian-cased')

sentences = ["Mario non è ancora arrivato.","Sono in ritardo."] # list of sentences

batch_encoded = tokenizer.batch_encode_plus(sentences, padding=True, add_special_tokens=True, return_tensors="pt")

with torch.no_grad():
  tokens_logits = model(**batch_encoded)

encodings = tokens_logits[0]
cls_encoding = encodings[:, 0, :]