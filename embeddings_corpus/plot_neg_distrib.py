import matplotlib.pyplot as pyplot
import numpy as np
import statistics as stat
import pandas as pd

print("Loading verb stats...")
verb_stats = pd.read_csv(f"new_database.csv")
# dataframe 2054 vbs w "lemma", "total occ", "num non neg", "num neg", "perc neg"
verb_stats = verb_stats.sort_values(by = "perc neg", ignore_index = True)



verbs_list = list(verb_stats["lemma"])
dictn = {}
n = 0
for vb in verbs_list:
    row = (verb_stats["lemma"] == vb)
    perc_neg = float(verb_stats.loc[row, "perc neg"])
    dictn[vb] = [n, perc_neg]
    n+=1

indexes = []
for vb in verbs_list:
    indexes.append(dictn[vb][0])


percs = []
for vb in verbs_list:
    percs.append(dictn[vb][1])
    


pyplot.scatter(indexes, percs, s=5)
pyplot.xticks(fontsize = 8, rotation = 30)
pyplot.title("Percentage of negated occurrences for all\n the verbs in the corpus, from least to most negated")
pyplot.xlabel("Rank", fontsize = 8)
pyplot.ylabel("Negated")
pyplot.savefig("plot_neg_perc.jpg")