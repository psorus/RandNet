from randnet import RandNet

from loaddata import loaddata

with open("../dex","r") as f:
    index=int(f.read().strip())

x,tx,ty=loaddata(index)

r=RandNet(lr=0.01,normalise=False)
auc=r.train(x,tx,ty)
with open("auc.json","w") as f:
    f.write(str(auc))
np.savez_compressed("auc.npz",auc=auc)



