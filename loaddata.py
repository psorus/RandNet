import numpy as np

def loaddata(key=0):
    #def filterdim(x,dims):
    #    return np.concatenate([np.expand_dims(x[d],axis=1) for d in dims],axis=1)
    print("before running any code, you should modify loaddata.py!")
    exit()
    fn="../../../data.npz"
    f=np.load(fn)
    x,y,tx,ty=f["train_x"],f["train_y"],f["test_x"],f["test_y"]
    x=np.reshape(x,(x.shape[0],28*28))
    tx=np.reshape(tx,(tx.shape[0],28*28))
    x=np.array([xx for xx,yy in zip(x,y) if yy==key])
    n=np.array([xx for xx,yy in zip(tx,ty) if yy==key])
    a=np.array([xx for xx,yy in zip(tx,ty) if yy!=key])
    tx=np.concatenate((n,a),axis=0)
    ty=np.array([0]*len(n)+[1]*len(a))
    x=x/255
    tx=tx/255
    return x,tx,ty


if __name__=="__main__":
    x,tx,ty=loaddata()
    print(x.shape,tx.shape,ty.shape)

