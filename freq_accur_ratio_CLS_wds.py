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
import matplotlib.pyplot as pyplot
import matplotlib
import statistics as stat
from math import log


freq = [1829124, 1352772, 1169009, 757341, 667545]
accur = [91.35, 90.45, 87.5, 83.0, 72.8]
#pyplot.bar([log(n) for n in freq], accur)
pyplot.bar(freq, accur, width=200)
#pyplot.scatter([log(n) for n in freq], accur)
matplotlib.pyplot.title("Variation of accuracy as a function of frequence")
pyplot.savefig("plot_classifier.png")