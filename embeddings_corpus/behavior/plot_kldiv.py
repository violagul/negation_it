from joblib import dump, load
import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean

kl_scores = load(r"kl_scores.joblib")
print(list(kl_scores.keys())[:10])
print(len(list(kl_scores.keys())))
print(type(kl_scores[list(kl_scores.keys())[0]]))


verb_stats = pd.read_csv(f"../new_database.csv")

verb_stats = verb_stats.sort_values(by = "perc neg", ascending = False, ignore_index = True)

#print(verb_stats.tail())


ranges = [(l, l+99) for l in range(0,2000, 100)]
ranges.append((2000, 2054))
#print(ranges)
lemmas = verb_stats["lemma"]
perc_neg = verb_stats["perc neg"]

verbs_rank_dict = {}
for rk in ranges:
    current_vb_list = []
    for n in range(rk[0], rk[1]):
        #if n<15:
            #print(n)
        current_vb_list.append(lemmas[n])
        #print(lemmas[n])
    verbs_rank_dict[rk] = current_vb_list

#print(len(verbs_rank_dict[(500, 599)]))


rk_kl_dict = {}
for rk in verbs_rank_dict.keys():
    current_kl_list = []
    for vb in verbs_rank_dict[rk]:
        if vb in kl_scores.keys():
            for elem in kl_scores[vb]:
                current_kl_list.append(elem)
    rk_kl_dict[rk] = current_kl_list

#for rk in ranges:
    #print(len(rk_kl_dict[rk]))
'''for elem in rk_kl_dict[(0, 99)]:
    print(len(elem))'''

mean_kl_dict = {}
means = []
lens = []
for rk in ranges:
    mean_kl_dict[rk] = mean([x.item() for x in rk_kl_dict[rk]])
    means.append(mean([x.item() for x in rk_kl_dict[rk]]))
    lens.append(len(rk_kl_dict[rk]))

ranks = list(range(len(ranges)))

plt.scatter(lens, means)
plt.savefig("kldiv_plot.jpg")