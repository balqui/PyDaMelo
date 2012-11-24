"""
Loads in a predictor, trains it with the data in
the datafile (in transactional form) and plots the
ROC curve for the chosen positive prediction pos_class

print_numbers controls whether a lot of numeric info
is dumped into the console

current notion of result.append tailored to NB
"""

import matplotlib.pyplot as plt

from naivebayes import NaiveBayes
from maxapost import MaxAPost
from data import Data

print_numbers = False

datafile = "../data/titanic.txt"
pos_class = "Survived:Yes"
##pos_class = "Survived:No"

##datafile = "weather.nominal.txt"
##pos_class = "play:yes"
##pos_class = "play:no"

##datafile = "haireyescolor.txt"
##pos_class = "Sex:Male"
##pos_class = "Sex:Female"

##datafile = "cmc-full.txt"
##pos_class = "contraceptive-method:none"

d = Data(datafile)

prnb = NaiveBayes(d)
##prnb = MaxAPost(d)
prnb.train()

if print_numbers:
    prnb.show()

pos = 0.0
neg = 0.0

for (v,c_true) in d.test_set:
    if c_true == pos_class: 
        pos += 1
    else:
        neg += 1

print

print "Predicting", pos_class, "for data file", datafile,
print "with", pos, "positive instances and", neg, "negative instances"

result = []
for (v,c_true) in d.test_set:
    """
    prepare predictions for sorting
    in case of equal weight, positive instances come first
    store both true class and first NB prediction
    """
    c_pred_nb = prnb.predict(v)
    wy = 0
    wn = 0
    for c in prnb.clssprobs:
        if c == pos_class: wy += prnb.value_weight(v,c)
        else: wn += prnb.value_weight(v,c)
    result.append((wy-wn,c_true==pos_class,
                   c_true,c_pred_nb[0]))
##    result.append((prnb.value_weight(v,pos_class),c_true==pos_class,
##                   c_true,c_pred_nb[0]))

if print_numbers:
    print "NB scores for predicting", pos_class
    for e in sorted(result): print e

    print "==="

trpos = 0
fapos = 0
x = [0.0]
y = [0.0]
if print_numbers:
    print "ROC curve for predicting", pos_class
    print fapos/neg, trpos/pos
for e in sorted(result,reverse=True):
    if e[2] == pos_class: 
        trpos += 1
    else: 
        fapos += 1
    if print_numbers:
        print fapos/neg, trpos/pos
    x.append(fapos/neg)
    y.append(trpos/pos)
plt.plot([-0.001,1.001],[-0.001,1.001],color="orange") # diagonal reference
plt.plot(x,y)
plt.show()

