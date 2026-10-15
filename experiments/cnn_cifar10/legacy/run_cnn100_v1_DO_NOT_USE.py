
#!/usr/bin/env python3
"""Resumable 100-seed CIFAR-10 federated backdoor stress-test harness.

Important: this is a transparent reimplementation based on the six-page manuscript's
described setup, NOT a byte-for-byte rerun of the unavailable original CNN script.
"""
import argparse, csv, json, random, time, os, hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pickle

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,64,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1)))
        self.classifier=nn.Linear(64,10)
    def forward(self,x): return self.classifier(self.features(x).flatten(1))

def poison(x,y,target=0,frac=.5,seed=0):
    rng=np.random.default_rng(seed); y=y.copy(); mask=np.where(y!=target)[0]
    n=int(round(len(mask)*frac)); chosen=rng.choice(mask,size=n,replace=False) if n else []
    x=x.copy()
    x[chosen,:,4:8,4:8]=1.0; y[chosen]=target
    return x,y

def stamp(x):
    z=x.clone(); z[:,:,4:8,4:8]=1.0; return z

def state_to_vec(state):
    return torch.cat([v.detach().cpu().reshape(-1).float() for v in state.values()])
def vec_to_state(vec, template):
    out={}; pos=0
    for k,v in template.items():
        n=v.numel(); out[k]=torch.as_tensor(vec[pos:pos+n].copy(), dtype=v.dtype).reshape(v.shape); pos+=n
    return out
def load_vec(model, vec): model.load_state_dict(vec_to_state(vec,model.state_dict()))

def train_local(global_vec, X, y, template, seed, device, epochs=1, batch_size=128, lr=.01, max_samples=0):
    if max_samples and len(y) > max_samples:
        rng=np.random.default_rng(seed); idx=rng.choice(len(y), size=max_samples, replace=False); X=X[idx]; y=y[idx]
    model=SmallCNN().to(device); load_vec(model,global_vec)
    ds=TensorDataset(torch.tensor(X,dtype=torch.float32),torch.tensor(y,dtype=torch.long))
    gen=torch.Generator().manual_seed(seed)
    loader=DataLoader(ds,batch_size=batch_size,shuffle=True,generator=gen)
    opt=torch.optim.SGD(model.parameters(),lr=lr,momentum=.9)
    model.train()
    for _ in range(epochs):
        for xb,yb in loader:
            xb,yb=xb.to(device),yb.to(device); opt.zero_grad()
            loss=nn.functional.cross_entropy(model(xb),yb); loss.backward(); opt.step()
    return state_to_vec(model.state_dict()).numpy()

@torch.no_grad()
def evaluate(vec,X,y,device,backdoor=False):
    model=SmallCNN().to(device); load_vec(model,vec); model.eval()
    xs=torch.tensor(X,dtype=torch.float32)
    if backdoor: xs=stamp(xs)
    correct=total=0
    for i in range(0,len(xs),512):
        xb=xs[i:i+512].to(device); pred=model(xb).argmax(1).cpu().numpy()
        yy=y[i:i+512]
        if backdoor:
            mask=yy!=0; correct+=int((pred[mask]==0).sum()); total+=int(mask.sum())
        else: correct+=int((pred==yy).sum()); total+=len(yy)
    return correct/max(total,1)

@torch.no_grad()
def probe_score(candidate_vec, Xroot, yroot, device):
    """Vectorized 3x3 spatial trigger probe for practical CPU execution.

    This is a documented approximation to the earlier 8x8 grid (64 positions).
    It evaluates nine spatial positions in one batched inference stream.
    """
    model=SmallCNN().to(device); load_vec(model,candidate_vec); model.eval()
    xs=torch.tensor(Xroot,dtype=torch.float32); ys=np.asarray(yroot)
    positions=[(0,0),(0,14),(0,28),(14,0),(14,14),(14,28),(28,0),(28,14),(28,28)]
    # Shape [positions, examples, channels, height, width], then flatten.
    patched=[]
    for r,c in positions:
        xp=xs.clone(); xp[:,:,r:r+4,c:c+4]=1.0; patched.append(xp)
    all_x=torch.cat(patched,dim=0)
    preds=[]
    with torch.no_grad():
        for i in range(0,len(all_x),256):
            preds.extend(model(all_x[i:i+256].to(device)).argmax(1).cpu().tolist())
    preds=np.asarray(preds).reshape(len(positions),len(xs))
    mask=ys!=0
    return float(np.max(np.mean(preds[:,mask]==0,axis=1)))

def aggregate(updates, method, scores=None, f=4, global_vec=None, root_gradient=None):
    U=np.stack(updates)
    if method=="fedavg": return U.mean(0),[],[]
    if method=="coord_median": return np.median(U,axis=0),[],[]
    if method=="trimmed_mean":
        trim=max(1,int(.2*len(U))); S=np.sort(U,axis=0)
        return S[trim:len(U)-trim].mean(0),[],[]
    if method=="mad":
        s=np.asarray(scores); med=np.median(s); mad=np.median(np.abs(s-med))
        flagged=np.where(s>med+max(2.0*mad,.1))[0].tolist()
        kept=[i for i in range(len(U)) if i not in flagged]
        if not kept: kept=list(np.argsort(s)[:len(U)-f])
        return U[kept].mean(0),flagged,kept
    if method=="topf":
        flagged=np.argsort(scores)[-f:].tolist(); kept=[i for i in range(len(U)) if i not in flagged]
        return U[kept].mean(0),flagged,kept
    if method=="oracle":
        # benchmark ceiling: remove the final f clients (malicious clients are assigned last)
        flagged=list(range(len(U)-f,len(U))); kept=list(range(len(U)-f))
        return U[kept].mean(0),flagged,kept
    if method=="multikrum":
        # Krum-style nearest-update selector; simplified implementation, explicitly labeled.
        d=((U[:,None,:]-U[None,:,:])**2).mean(2)
        scores_k=np.sort(d,axis=1)[:,1:max(2,len(U)-f-2)].sum(1)
        chosen=np.argsort(scores_k)[:len(U)-f]
        return U[chosen].mean(0),[i for i in range(len(U)) if i not in chosen],chosen.tolist()
    if method=="fltrust":
        # Trust by cosine similarity to clean root-set gradient; simplified reference.
        if root_gradient is None: return U.mean(0),[],[]
        norm=np.linalg.norm(root_gradient)+1e-12
        trust=np.maximum(0,(U@root_gradient)/(np.linalg.norm(U,axis=1)*norm+1e-12))
        trust=trust/(trust.sum()+1e-12)
        return (trust[:,None]*U).sum(0),[],list(range(len(U)))
    raise ValueError(method)

def root_gradient(global_vec,X,y,device):
    model=SmallCNN().to(device); load_vec(model,global_vec); model.train()
    xb=torch.tensor(X,dtype=torch.float32).to(device); yb=torch.tensor(y,dtype=torch.long).to(device)
    model.zero_grad(); nn.functional.cross_entropy(model(xb),yb).backward()
    return state_to_vec({k:p.grad for (k,p) in model.named_parameters()}).numpy()

def run_one(seed, alpha, args, Xtrain,Ytrain,Xtest,Ytest):
    set_seed(seed); rng=np.random.default_rng(seed)
    device=args.device
    # fixed 1000-image server-root sample, stratification not assumed
    idx=rng.choice(len(Xtrain),size=args.root_size,replace=False)
    Xroot,Yroot=Xtrain[idx],Ytrain[idx]
    remain=np.setdiff1d(np.arange(len(Xtrain)),idx)
    # Dirichlet label-skew partition with 20 clients
    client_ids=[[] for _ in range(args.clients)]
    for cls in range(10):
        ids=remain[Ytrain[remain]==cls]; rng.shuffle(ids)
        props=rng.dirichlet(np.full(args.clients,alpha))
        cuts=(np.cumsum(props)*len(ids)).astype(int)[:-1]
        for i,part in enumerate(np.split(ids,cuts)): client_ids[i].extend(part.tolist())
    # ensure no empty clients
    for i in range(args.clients):
        if not client_ids[i]: client_ids[i]=rng.choice(remain,size=32,replace=True).tolist()
    Xc=[];Yc=[]
    for i,ids in enumerate(client_ids):
        x,y=Xtrain[ids],Ytrain[ids]
        if i>=args.clients-args.malicious:
            x,y=poison(x,y,frac=.5,seed=seed*100+i)
        Xc.append(x);Yc.append(y)
    model=SmallCNN().to(device); global_vec=state_to_vec(model.state_dict()).numpy()
    results=[]
    for method in args.methods:
        w=global_vec.copy(); total_tp=total_fp=total_m=total_b=0
        for rnd in range(args.rounds):
            updates=[]; scores=[]
            for i in range(args.clients):
                local=train_local(w,Xc[i],Yc[i],model.state_dict(),seed*100000+rnd*args.clients+i,device,epochs=args.local_epochs,batch_size=args.batch_size,lr=args.lr,max_samples=args.max_client_samples)
                updates.append(local-w)
                if method in ("mad","topf"):
                    scores.append(probe_score(w+updates[-1],Xroot,Yroot,device))
            # root gradient calculated on current model (for simplified FLTrust)
            rg=root_gradient(w,Xroot,Yroot,device) if method=="fltrust" else None
            delta,flagged,kept=aggregate(updates,method,scores if scores else None,f=args.malicious,global_vec=w,root_gradient=rg)
            w=w+delta
            if method in ("mad","topf"):
                malicious=set(range(args.clients-args.malicious,args.clients))
                total_tp+=len(malicious.intersection(flagged)); total_fp+=len(set(flagged)-malicious)
                total_m+=args.malicious; total_b+=args.clients-args.malicious
        acc=evaluate(w,Xtest,Ytest,device,False); asr=evaluate(w,Xtest,Ytest,device,True)
        results.append(dict(seed=seed,alpha=alpha,method=method,asr=asr,clean_accuracy=acc,
            cumulative_tpr=(total_tp/total_m if total_m else float("nan")),
            cumulative_fpr=(total_fp/total_b if total_b else float("nan")),
            rounds=args.rounds,clients=args.clients,malicious=args.malicious,root_size=args.root_size,
            implementation="reimplementation_not_original"))
    return results

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",default="./data"); ap.add_argument("--out",default="./results")
    ap.add_argument("--seeds",default="0:100"); ap.add_argument("--alphas",default="0.5,1,10")
    ap.add_argument("--methods",default="fedavg,coord_median,trimmed_mean,multikrum,fltrust,mad,topf,oracle")
    ap.add_argument("--clients",type=int,default=20); ap.add_argument("--malicious",type=int,default=4)
    ap.add_argument("--rounds",type=int,default=15); ap.add_argument("--local-epochs",type=int,default=1)
    ap.add_argument("--root-size",type=int,default=200); ap.add_argument("--batch-size",type=int,default=128)
    ap.add_argument("--lr",type=float,default=.01); ap.add_argument("--max-client-samples",type=int,default=0,help="optional cap per client per local epoch; 0 uses all assigned data"); ap.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    args=ap.parse_args(); torch.set_num_threads(1); torch.set_num_interop_threads(1); args.methods=args.methods.split(",")
    a,b=map(int,args.seeds.split(":")); seeds=range(a,b); alphas=[float(x) for x in args.alphas.split(",")]
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True); raw=out/"cnn100_raw.csv"
    # Load the user-supplied official CIFAR-10 python archive locally (no network required).
    base=Path(args.data)
    if (base / "cifar-10-batches-py").exists(): base=base / "cifar-10-batches-py"
    def read_batch(path):
        with open(path, "rb") as f: d=pickle.load(f, encoding="bytes")
        return np.asarray(d[b"data"], dtype=np.uint8), np.asarray(d[b"labels"], dtype=np.int64)
    train_parts=[read_batch(base / f"data_batch_{i}") for i in range(1,6)]
    tr=np.concatenate([z[0] for z in train_parts]); Ytrain=np.concatenate([z[1] for z in train_parts])
    Xtrain=tr.reshape(-1,3,32,32).astype(np.float32)/255.0
    te,Ytest=read_batch(base / "test_batch")
    Xtest=te.reshape(-1,3,32,32).astype(np.float32)/255.0
    existing=[]
    if raw.exists():
        with raw.open() as f: existing=list(csv.DictReader(f))
    done={(int(r["seed"]),float(r["alpha"]),r["method"]) for r in existing}
    for alpha in alphas:
        for seed in seeds:
            if all((seed,alpha,m) in done for m in args.methods): 
                print(f"skip completed seed={seed} alpha={alpha}", flush=True)
                continue
            start=time.time()
            rows=run_one(seed,alpha,args,Xtrain,Ytrain,Xtest,Ytest)
            with raw.open("a",newline="") as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0])); 
                if raw.stat().st_size==0: writer.writeheader()
                writer.writerows([r for r in rows if (r["seed"],r["alpha"],r["method"]) not in done])
            print(f"finished seed={seed} alpha={alpha} methods={len(rows)} elapsed={time.time()-start:.1f}s",flush=True)
    print(f"raw results: {raw}")
if __name__=="__main__": main()
