"""
Trains a 2-class NB or MAP predictor with the data in
the datafile (in transactional form) and plots the
ROC curve for each class and for the difference.


"""


import matplotlib.pyplot as plt


from naivebayes import NaiveBayes
from maxapost import MaxAPost
from data import Data


datafile = "lenses-tr.txt"
pos_class = "none"

##datafile = "../data/titanic.txt"
##pos_class = "Survived:No"
##pos_class = "Survived:Yes"

##datafile = "../data/weather.nominal.txt"
##pos_class = "play:yes"
##pos_class = "play:no"

##datafile = "haireyescolor.txt"
##pos_class = "Sex:Male"
##pos_class = "Sex:Female"

##datafile = "../data/cmc-full.txt"
##pos_class = "contraceptive-method:none"
##pos_class = "contraceptive-method:long-term"
##pos_class = "contraceptive-method:short-term"

d = Data(datafile)
prnb = NaiveBayes(d)
##prnb = MaxAPost(d)
prnb.train()

pos = 0.0
neg = 0.0

for (v,c_true) in d.test_set:
    if c_true == pos_class:
        pos += 1
    else:
        neg += 1

result_pos = []
result_neg = []
result_dif = []
result_nor = []
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
    result_dif.append((wy-wn,c_true==pos_class,
                       c_true,c_pred_nb[0]))
    result_pos.append((wy,c_true==pos_class,
                       c_true,c_pred_nb[0]))
    result_neg.append((wn,c_true!=pos_class,
                       c_true,c_pred_nb[0]))
    result_nor.append((wy/(wy+wn),c_true==pos_class,
                       c_true,c_pred_nb[0]))

plt.plot([-0.001,1.001],[-0.001,1.001],color="orange") # diagonal reference

trpos = 0
fapos = 0
x = [0.0]
y = [0.0]
for e in sorted(result_dif,reverse=True):
    if e[2] == pos_class:
        trpos += 1
    else:
        fapos += 1
    x.append(fapos/neg)
    y.append(trpos/pos)
plt.plot(x,y) # green

trpos = 0
fapos = 0
x = [0.0]
y = [0.0]
for e in sorted(result_pos,reverse=True):
    if e[2] == pos_class:
        trpos += 1
    else:
        fapos += 1
    x.append(fapos/neg)
    y.append(trpos/pos)
plt.plot(x,y) # red

trpos = 0
fapos = 0
x = [0.0]
y = [0.0]
for e in sorted(result_neg):
    if e[2] == pos_class:
        trpos += 1
    else:
        fapos += 1
    x.append(fapos/neg)
    y.append(trpos/pos)
plt.plot(x,y) # cyan

trpos = 0
fapos = 0
x = [0.0]
y = [0.0]
for e in sorted(result_nor,reverse=True):
    if e[2] == pos_class:
        trpos += 1
    else:
        fapos += 1
    x.append(fapos/neg)
    y.append(trpos/pos)
plt.plot(x,y) # purple

plt.show()

