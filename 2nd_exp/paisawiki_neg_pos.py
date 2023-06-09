with open(r"paisa.raw.utf8", encoding='utf8') as infile:
#with open(r"paisa.raw.utf8") as infile:
    
    paisa = infile.read()

import re

wiki_pattern = r"<text.*wiki.*(?:\n.*)+?\n</text>\n" # finds all texts containing "wiki" in their tag's url

paisa_wiki = re.findall(wiki_pattern, paisa)

print(f"Number of texts from a site containing 'wiki' in their URL: {len(paisa_wiki)}")

sent = []

pattern = r" [A-Z][a-z ]*[,:]?[a-z ]+[,:]?[a-z ][,:]?[a-z]+\. \b"  # finds kind of acceptable sentences



for text in paisa_wiki:
  found = re.findall(pattern, text)
  for elem in  found:
    if len(elem) > 25:
      sent.append(elem)



print(f"Number of sentences: {len(sent)}")

sent_pos = []
sent_neg = []

neg_patt = r"\b[Nn]on\b"  # finds the negation in a sentence



for s in sent:
  matches = re.search(neg_patt, s)
  if matches:
    sent_neg.append(s)
  else:
    sent_pos.append(s)



print(f"Number of positive sentences : {len(frasi_pos)}\n\nNumber of negative sentences : {len(frasi_neg)}")