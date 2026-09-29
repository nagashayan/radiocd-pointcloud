import argparse,time,numpy as np,pandas as pd,tensorflow as tf
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from tensorflow import keras
from tensorflow.keras import layers

FM={"xyz":[0,1,2],"xyzd":[0,1,2,3],"xyzi":[0,1,2,4],"xyzdi":[0,1,2,3,4],"i":[4]}

def block(x,n):
    x=layers.Conv1D(n,1,use_bias=False)(x); x=layers.BatchNormalization()(x)
    return layers.Activation("relu")(x)

def build(npnt,nd):
    i=keras.Input((npnt,nd)); x=i
    for n in [64,64,64,128,256]: x=block(x,n)
    x=layers.GlobalMaxPooling1D()(x)
    for n in [256,128]:
        x=layers.Dense(n,use_bias=False)(x); x=layers.BatchNormalization()(x)
        x=layers.Activation("relu")(x); x=layers.Dropout(.3)(x)
    o=layers.Dense(5,activation="softmax")(x)
    m=keras.Model(i,o); m.compile(keras.optimizers.Adam(1e-3),"sparse_categorical_crossentropy",["accuracy"])
    return m

ap=argparse.ArgumentParser()
ap.add_argument("--data",type=Path,default=Path("outputs/pointnet_xyzdi_32.npz"))
ap.add_argument("--features", choices=list(FM.keys()), default="xyzi")
ap.add_argument("--fold",type=int,default=1)
ap.add_argument("--output-dir",type=Path,default=Path("outputs/pointnet_ablation"))
a=ap.parse_args(); tf.keras.utils.set_random_seed(12)
d=np.load(a.data,allow_pickle=True)
X=d["X"][:,:,FM[a.features]].astype(np.float32)
y=d["y"].astype(np.int64)
csv=d["csv"].astype(str)

print("X dtype:", X.dtype, "y dtype:", y.dtype)
u=np.unique(csv); tr,te=list(KFold(10,shuffle=True,random_state=12).split(u))[a.fold-1]
tm=np.isin(csv,u[tr]); vm=np.isin(csv,u[te])
flat=X[tm].reshape(-1,X.shape[-1]); mean,std=flat.mean(0),flat.std(0); std[std==0]=1
xt=((X[tm]-mean)/std).astype("float32"); xv=((X[vm]-mean)/std).astype("float32")
out=a.output_dir; (out/"checkpoints").mkdir(parents=True,exist_ok=True); ck=out/"checkpoints"/f"{a.features}_fold{a.fold}.weights.h5"
m=build(X.shape[1],X.shape[2]); params=m.count_params()
cb=[keras.callbacks.EarlyStopping(monitor="val_accuracy",patience=30,mode="max"),
keras.callbacks.ModelCheckpoint(str(ck),monitor="val_accuracy",mode="max",save_best_only=True,save_weights_only=True),
keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=.5,patience=10,min_lr=1e-5)]
print("Features:",a.features,"Shape:",X.shape,"Parameters:",params)
t=time.perf_counter(); h=m.fit(xt,y[tm],validation_data=(xv,y[vm]),batch_size=128,epochs=300,verbose=0,callbacks=cb); secs=time.perf_counter()-t
m.load_weights(ck); t=time.perf_counter(); pred=m.predict(xv,batch_size=512,verbose=0).argmax(1); infer=time.perf_counter()-t
acc=float((pred==y[vm]).mean()); f1=float(f1_score(y[vm],pred,average="macro")); epochs=len(h.history["loss"]); best=int(np.argmax(h.history["val_accuracy"])+1)
r={"features":a.features,"fold":a.fold,"accuracy":acc,"macro_f1":f1,"epochs_ran":epochs,"best_epoch":best,"train_seconds":secs,"seconds_per_epoch":secs/epochs,"inference_ms_per_sample":infer/vm.sum()*1000,"parameters":params}
pd.DataFrame([r]).to_csv(out/f"result_{a.features}_fold{a.fold}.csv",index=False)
print(f"Accuracy={acc:.4f} MacroF1={f1:.4f} Epochs={epochs} BestEpoch={best}")
print(f"Training={secs/60:.1f} min, {secs/epochs:.2f} s/epoch; inference={infer/vm.sum()*1000:.4f} ms/sample")
