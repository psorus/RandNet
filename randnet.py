import numpy as np

import tensorflow as tf
from tensorflow import keras
from densedrop import densedrop

from sklearn.metrics import roc_auc_score

import os


class RandNet():
    def __init__(self, alpha=0.5,lr=0.01,pth="results/",normalise=True):
        self.alpha=alpha
        self.lr=lr
        self.bpth=pth
        self.normalise=normalise

    def normalize(self, x):
        if not self.normalise:return x
        return (x-self.mn)/(self.mx-self.mn)
    
    def gen_dims(self,inpd):
        dims=[self.alpha,self.alpha**2,self.alpha**3,self.alpha**2,self.alpha,1]
        dims=[max([3,int(d*inpd)]) for d in dims]
        return dims

    def _train_one(self,x,tx,ty,dex):
        pth=self.bpth+str(dex)+"/"
        os.makedirs(pth,exist_ok=True)

        def genlayer(dim,act="relu"):
            return densedrop(dim,activation=act),keras.layers.Dense(dim,activation=act)
        def genlayers(dims):
            layers=[]
            for dim in dims[:-1]:
                layers.append(genlayer(dim))
                layers.append(genlayer(dims[-1],act="linear"))
            return layers
        def genmodel(inp,inp2,layers):
            q=inp
            q2=inp2
            for layer in layers:
                q=layer[0](q)
                q2=layer[1](q2)
            return keras.models.Model(inp,q),keras.models.Model(inp2,q2)

        inp=keras.layers.Input(shape=x.shape[1:])
        inp2=keras.layers.Input(shape=x.shape[1:])
        dims=self.gen_dims(int(x.shape[1]))
        layers=genlayers(dims)
        model,model2=genmodel(inp,inp2,layers)
        model.summary()

        model.compile(optimizer=keras.optimizers.Adam(lr=self.lr),
                  loss=keras.losses.mean_squared_error)
        model.summary()
        model2.compile(optimizer=keras.optimizers.Adam(lr=self.lr),
                  loss=keras.losses.mean_squared_error)
        model2.summary()

        model.fit(x,x,
                epochs=300,
                batch_size=100,
                validation_split=0.1,
                verbose=1,
                shuffle=True)

        
        px= model.predict(x)
        dx=(px-x)**2
        dx=np.mean(dx,axis=1)
        div=np.std(dx)
    
        wei=model.get_weights()
        alts=[lay[0].get_matrix() for lay in layers]
        for i,w in enumerate(wei):
            if str(alts[0].shape)==str(w.shape):
                wei[i]=alts.pop(0)
                if len(alts)==0:
                    break
    
        model2.set_weights(wei)
    
        p=model2.predict(tx)
        d=(p-tx)**2
        d=np.mean(d,axis=1)
    
    
        auc=roc_auc_score(ty,d)
        print(auc)

        model2.save(f"{pth}/model.h5")
        model2.save(f"{pth}/saved")
        np.savez_compressed(f"{pth}/result.npz",y_true=ty,y_score=d,div=div)
        with open(f"{pth}/auc","w") as f:
            f.write(str(auc))
        return model2,d,div,ty,auc


    def train(self,x,tx,ty):
        self.mn,self.mx=np.min(x,axis=0),np.max(x,axis=0)
        x=self.normalize(x)
        tx=self.normalize(tx)
        self.ds=[]
        for i in range(200):
            print("training ",i)
            _,d,div,_,_=self._train_one(x,tx,ty,i)
            self.ds.append(d/div)
        self.ds=np.array(self.ds)
        self.y_score=np.median(self.ds**2,axis=0)
        self.auc=roc_auc_score(ty,self.y_score)
        print("AUC: ",self.auc)
        return self.auc














