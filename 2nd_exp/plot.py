import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import numpy as np
import matplotlib.pyplot as pyplot
import matplotlib
import statistics as stat
from math import log


w=0.1
contxt = ["CpTn", "CnTp", "CnTn", "CpTv"]
bar1 = np.arange(len(contxt))
bar2 = [i+w for i in bar1]
bar3 = [i+w for i in bar2]
bar4 = [i+w for i in bar3]
bar5 = [i+w for i in bar4]

bert_it = [83.55, 88.97, 88.42, 98.72]
mbert = [77.03, 80.34, 81.37, 99.03]
alberto = [90.33, 95.64, 90.66, 99.85]
umberto = [90.09, 88.13, 79.42, 98.35]
bertxxl = [77.87, 85.46, 85.40, 85.66]

pyplot.bar(bar1, bert_it,w ,label = "BERT italian")
pyplot.bar(bar2, mbert, w,label = "mBERT")
pyplot.bar(bar3, alberto,w ,label = "AlBERTo")
pyplot.bar(bar4, umberto,w ,label = "UmBERTo")
pyplot.bar(bar5, bertxxl,w ,label = "BERT xxl")
pyplot.xticks(bar3, contxt)
pyplot.ylabel("Percentage of repetition")

matplotlib.pyplot.title("Repetitions")
pyplot.legend(loc='upper left', ncols=3, fontsize=7)

pyplot.savefig("plot_templ.png")