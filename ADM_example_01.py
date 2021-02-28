from data import Data
from naivebayes import NaiveBayes

filename = "datasets/weatherNominal.td"
## filename = "datasets/titanic.td"
## filename = "datasets/cmc.td"

d = Data(filename)
d.report()

pr = NaiveBayes(d)
pr.train()
pr.show()

for (v,c_true) in d.test_set:
        c_pred = pr.predict(v)[0]
        print(v, ":")
        print("   ", c_pred, "( true class:", c_true, ")")

##    print(pr.predict(("Class:1st","Sex:Female","Age:Child")))

##    print(pr.predict(("Class:Crew","Sex:Female","Age:Child")))
