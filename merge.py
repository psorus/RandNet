import numpy as np
import os

from sklearn.metrics import roc_auc_score
from simplestat import statinf

import json

fns=[f"results/{zw}/result.npz" for zw in os.listdir("results")]
fns=[fn for fn in fns if os.path.isfile(fn)]



y_true=None
y_scores=[]
for fn in fns:
    f=np.load(fn)
    if y_true is None:
        y_true=f["y_true"]
    y_scores.append(f["y_score"]/f["div"])



aucs=[roc_auc_score(y_true,y_score) for y_score in y_scores]

print(json.dumps(statinf(aucs),indent=2))

y_score=np.median(np.abs(y_scores)**2,axis=0)
auc=roc_auc_score(y_true,y_score)

print("---total---:",auc)

c=os.getcwd()
c=c[c.rfind("/"):]


with open(f"../stats/{c}",'w') as f:
    f.write(str(auc))

