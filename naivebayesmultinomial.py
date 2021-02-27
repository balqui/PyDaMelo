
"""
Naive Bayes on transactional data - multinomial handling

No Laplace for the time being
"""
from collections import defaultdict
#from data_noheader import Data # ????????
from data import Data

class NaiveBayesMultinomial:

    def __init__(self,data):
        self.data = data
        self.normalized = False
        self.clsscnts = defaultdict(int)
        self.totalcls = defaultdict(int)
        self.itemscls = defaultdict(int)
        self.condprobs = {}
        self.clssprobs = {}

    def train(self):
        "store counts of appearing items"
        self.normalized = False
        for (v,c) in self.data.training_set:
            self.clsscnts[c] += 1
            for item in v:
                self.totalcls[c] += 1
                self.itemscls[item,c] += 1
        self.normalize()

    def normalize(self):
        if self.normalized: return
        self.normalized = True
        for item, c in self.itemscls:
            self.condprobs[item,c] = \
                float(self.itemscls[item,c])/self.totalcls[c]
        for c in self.clsscnts:
            self.clssprobs[c] = float(self.clsscnts[c])/self.data.N

    def value_weight(self,items,clval):
        "weight of class value clval for present and absent items"
        prc = self.clssprobs[clval]
        for item in items:
            prc *= self.condprobs[item,clval]
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
            for item, cls in sorted(self.itemscls):
                if c == cls:
                    print(item, self.condprobs[item,c])

if __name__=="__main__":

    filename = \
    "datasets/markbaskhome.td"
#    "datasets/markbasksex.td"
    d = Data(filename)

    d.report()
    
    pr = NaiveBayesMultinomial(d)
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

##    print pr.predict(("Class:1st","Sex:Female","Age:Child"))

##    print pr.predict(("Class:Crew","Sex:Female","Age:Child"))
