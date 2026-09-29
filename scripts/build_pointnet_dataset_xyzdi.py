from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from tqdm import tqdm

WINDOW, THRESHOLD_FRAMES, THRESHOLD_GAP, EPS = 10, 8, 2, 0.5
CHANNELS = ["X", "Y", "Z", "Doppler", "Intensity"]

def fixed_count(a, n, rng):
    if len(a) > n: return a[rng.choice(len(a), n, replace=False)]
    if len(a) < n: return np.concatenate([a, a[rng.choice(len(a), n-len(a), replace=True)]])
    return a

def process(path, root, n_points, rng):
    df = pd.read_csv(path)
    if "Presence" not in df.columns: return [], True
    parts = path.relative_to(root).parts
    subject, env, cls = parts[0], parts[1], parts[2]
    pf = np.sort(df.loc[df["Presence"] == 1, "Frame #"].unique())
    if len(pf) < 8: return [], False
    pfset, allf = set(pf), set(df["Frame #"].unique())
    out=[]
    for start in pf:
        fw=np.arange(start,start+10)
        if len(set(fw)-pfset)>2 or fw[-1] not in allf: continue
        cur=df[(df["Frame #"]>=fw[0])&(df["Frame #"]<=fw[-1])]
        moving=cur[(cur["Presence"]==1)&(cur["Doppler"].abs()>0)].copy()
        if len(moving)<8: continue
        xyz=moving[["X","Y","Z"]].to_numpy(np.float32)
        labels=DBSCAN(eps=0.5,min_samples=8).fit_predict(xyz)
        best=None; bestdist=np.inf
        for lab in np.unique(labels):
            if lab == -1: continue
            center=xyz[labels==lab].mean(axis=0)
            dist=float(np.linalg.norm(center))
            if abs(center[0])<0.3 and 0<dist<bestdist:
                best,bestdist=lab,dist
        if best is None: continue
        chosen=moving.iloc[np.where(labels==best)[0]]
        if not (1.0 < chosen["Y"].mean() < 2.5 and chosen["Z"].mean() < 1.0): continue
        pts=chosen[CHANNELS].to_numpy(np.float32)
        original_n=len(pts)
        pts=fixed_count(pts,n_points,rng).astype(np.float32)
        out.append((pts,cls,subject,env,str(path),int(fw[0]),int(fw[-1]),original_n))
    return out, False

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("dataset_root",type=Path)
    ap.add_argument("--output",type=Path,default=Path("outputs/pointnet_xyzdi_32.npz"))
    ap.add_argument("--n-points",type=int,default=32)
    ap.add_argument("--seed",type=int,default=12)
    a=ap.parse_args()
    root=a.dataset_root.expanduser().resolve()
    rng=np.random.default_rng(a.seed)
    clouds=[]; labels=[]; subjects=[]; envs=[]; csvs=[]; starts=[]; ends=[]; counts=[]
    skipped=0; errors=[]
    files=sorted(root.rglob("*.csv"))
    for path in tqdm(files,desc="Processing recordings"):
        try:
            samples,sk=process(path,root,a.n_points,rng); skipped+=int(sk)
            for x,c,s,e,f,st,en,n in samples:
                clouds.append(x); labels.append(c); subjects.append(s); envs.append(e)
                csvs.append(f); starts.append(st); ends.append(en); counts.append(n)
        except Exception as exc: errors.append((str(path),repr(exc)))
    X=np.stack(clouds).astype(np.float32)
    names=np.array(["backpack","chair","desk","human","wall"])
    cmap={v:i for i,v in enumerate(names)}
    y=np.array([cmap[v] for v in labels],dtype=np.int64)
    output=a.output.expanduser().resolve(); output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,X=X,y=y,class_names=names,channels=np.array(CHANNELS),
        subject=np.array(subjects),environment=np.array(envs),csv=np.array(csvs),
        start_frame=np.array(starts),end_frame=np.array(ends),original_n_points=np.array(counts))
    print(f"\nSamples: {len(X)}\nShape: {X.shape}\nSkipped without Presence: {skipped}\nErrors: {len(errors)}")
    for name,i in cmap.items(): print(f"{name:10s}: {(y==i).sum()}")
    print(f"Saved: {output}")
    if errors: pd.DataFrame(errors,columns=["file","error"]).to_csv(output.with_suffix(".errors.csv"),index=False)

if __name__=="__main__": main()
