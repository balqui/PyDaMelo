
"""
Naive Bayes on transactional data - standard handling (= Bernouilli)

No Laplace for the time being
"""
from collections import defaultdict
##from data_noheader import Data   # AVERIGUAR QUE ERA data_noheader
from data import Data

class NaiveBayesTrans:

    def __init__(self,data):
        self.data = data
        self.normalized = False
        self.clsscnts = defaultdict(int)
        self.condcnts = defaultdict(int)
        self.itemcnts = defaultdict(int) # guess a set would suffice
        self.condprobs = {}
        self.clssprobs = {}

    def train(self):
        "store counts of appearing items"
        self.normalized = False
        for (v,c) in self.data.training_set:
            self.clsscnts[c] += 1
            for item in set(v):
                "set removes duplicates, just in case"
                self.itemcnts[item] += 1
                self.condcnts[item,c] += 1
        self.normalize()

    def normalize(self):
        if self.normalized: return
        self.normalized = True
        for item in self.itemcnts:
            for c in self.clsscnts:
                self.condprobs[item,c] = \
                    float(self.condcnts[item,c])/self.clsscnts[c]
        for c in self.clsscnts:
            self.clssprobs[c] = float(self.clsscnts[c])/self.data.N

    def value_weight(self,items,clval):
        "weight of class value clval for present and absent items"
        prc = self.clssprobs[clval]
        for item in self.itemcnts:
            if item in items:
                prc *= self.condprobs[item,clval]
            else:
                prc *= 1 - self.condprobs[item,clval]
        return prc

# missing method float value prediction

    def predict(self,items):
        predictions = []
        mx = 0.0
        for c in self.clssprobs.keys():
            prc = self.value_weight(items,c)
            if prc > mx:
                predictions = []
            if prc >= mx:
                mx = prc
                predictions.append(c)
        return predictions

    def show(self):
        print("N =", self.data.N)
        print("\nclass probs:")
        for c in self.clssprobs:
            print(c, self.clssprobs[c])
        print("\nitem probs:")
        for c in self.clssprobs:
            print("\nclass", c, ":")
            for a in sorted(self.itemcnts):
                print(a, self.condprobs[a,c])

if __name__=="__main__":

    filename = \
    "datasets/markbaskhome.txt"
#    "datasets/markbasksex.txt"
    d = Data(filename)

    d.report()
    
    pr = NaiveBayesTrans(d)
    pr.train()
    pr.show()
#    exit()
    


    from confmat import ConfMat
    
    cm = ConfMat(pr.clsscnts)
    print()
    for (v,c_true) in d.test_set:
        c_pred = pr.predict(v)[0]
#        print(v, c_pred, "( true class:", c_true, ")")
        cm.mat[c_pred,c_true] += 1
    print()
    pr.show()
    print()
    cm.report()        

##    print(pr.predict(("Class:1st","Sex:Female","Age:Child")))

##    print(pr.predict(("Class:Crew","Sex:Female","Age:Child")))
