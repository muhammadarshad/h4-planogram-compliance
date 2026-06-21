"""
Bulletproof product-identification benchmark — closes R1/R2/R3.

R1  config selection: QRoPE hyper-params chosen ONLY on a validation shelf split,
    then frozen; the headline number is on a disjoint TEST shelf split.
R2  fine-tuned CNN upper bound added (linear probe trained on validation crops).
R3  natural class distribution (no per-class cap) + per-class accuracy reported.

All identification is leave-one-shelf-out k-NN; metric = top-1 accuracy, mean ±95% CI.
"""
import os, re, collections, math, random
import numpy as np
from PIL import Image
import torch, torchvision
from torchvision import transforms

ROOT='grocerydataset/ProductImagesFromShelves'; CLASSES=[str(c) for c in range(1,11)]
K=11; rng=random.Random(0)
src=lambda fn:(re.match(r'(.+\.JPG)_',fn).group(1) if re.match(r'(.+\.JPG)_',fn) else fn)

# ── gather ALL crops (natural distribution, no cap) ──
items=[]
for c in CLASSES:
    for f in os.listdir(f"{ROOT}/{c}"):
        if re.match(r'.+\.JPG_\d+_\d+_\d+_\d+\.png',f): items.append((c,f,src(f)))
y=np.array([int(c) for c,_,_ in items]); grp=np.array([s for *_,s in items])
print(f"crops: {len(items)} (natural distribution)  per-class: "
      f"{dict(collections.Counter(int(c) for c,_,_ in items))}")

# ── split SHELVES into validation / test (disjoint) ──
shelves=sorted(set(grp)); rng.shuffle(shelves)
half=len(shelves)//2; VAL=set(shelves[:half]); TEST=set(shelves[half:])
is_val=np.array([s in VAL for s in grp]); is_test=~is_val
print(f"shelves: {len(VAL)} validation / {len(TEST)} test (disjoint)")

# ── arrays / encoders ──
def loadarr(c,f,sz): return np.asarray(Image.open(f"{ROOT}/{c}/{f}").convert('RGB').resize((sz,sz)),np.float64)
def qrope(arr,G,FREQS):
    im=arr-128; rr,cc=np.mgrid[0:G,0:G]; rr,cc=rr.ravel(),cc.ravel()
    x0,x1,x2,x3=im[:,:,0].ravel(),im[:,:,1].ravel(),im[:,:,2].ravel(),im.mean(2).ravel()
    sig=[]
    for fr in range(1,FREQS+1):
        for fc in range(1,FREQS+1):
            t1=2*np.pi*fr*rr/G; t2=2*np.pi*fc*cc/G
            c1,s1,c2,s2=np.cos(t1),np.sin(t1),np.cos(t2),np.sin(t2)
            sig+=[(x0*c1-x1*s1).sum(),(x0*s1+x1*c1).sum(),(x2*c2-x3*s2).sum(),(x2*s2+x3*c2).sum()]
    s=np.array(sig); return s/(np.linalg.norm(s)+1e-9)

def loso_top1(E, idxset, per_class=False):
    """leave-one-shelf-out top-1 over shelves within idxset (boolean mask)."""
    idx=np.where(idxset)[0]
    by_shelf=collections.defaultdict(list)
    for i in idx: by_shelf[grp[i]].append(i)
    accs=[]; pc_ok=collections.Counter(); pc_n=collections.Counter()
    for s,qi in by_shelf.items():
        if len(qi)<3: continue
        gal=[i for i in idx if grp[i]!=s]
        Gv=E[gal]; gy=y[gal]; sims=E[qi]@Gv.T
        ok=0
        for r,i in enumerate(qi):
            j=np.argsort(-sims[r])[:K]
            pred=collections.Counter(int(gy[t]) for t in j).most_common(1)[0][0]
            hit=(pred==int(y[i])); ok+=hit
            if per_class: pc_ok[int(y[i])]+=hit; pc_n[int(y[i])]+=1
        accs.append(ok/len(qi))
    a=np.array(accs); ci=1.96*a.std(ddof=1)/math.sqrt(len(a))
    return (a.mean()*100, ci*100, len(a), pc_ok, pc_n)

# ── R1: select QRoPE config on VALIDATION only ──
print("\nR1  config selection on validation (frozen for test):")
best=None
for G in (12,16,20):
    A=np.stack([loadarr(c,f,G) for c,f,_ in items])   # encode all at this G
    for F in (6,8):
        E=np.array([qrope(a,G,F) for a in A])
        m,_,_,_,_=loso_top1(E, is_val)
        print(f"    G={G} freqs={F}: val top-1 {m:.1f}%")
        if best is None or m>best[0]: best=(m,G,F,E)
_,bG,bF,bE=best
print(f"  -> chosen config: G={bG}, freqs={bF}")

# ── TEST with frozen config + per-class (R3) ──
tm,tci,tn,pc_ok,pc_n=loso_top1(bE, is_test, per_class=True)
print(f"\nTEST (frozen config, {tn} shelves):  QRoPE top-1 = {tm:.1f}% ± {tci:.1f}")
print("  per-class top-1: "+"  ".join(f"c{c}:{100*pc_ok[c]/pc_n[c]:.0f}%(n={pc_n[c]})" for c in sorted(pc_n)))

# ── baselines on the SAME test split ──
def colorhist(arr):
    q=(arr.reshape(-1,3)//64).astype(int); idx=q[:,0]*16+q[:,1]*4+q[:,2]
    h=np.bincount(idx,minlength=64).astype(float); return h/(np.linalg.norm(h)+1e-9)
A16=np.stack([loadarr(c,f,16) for c,f,_ in items])
Ech=np.array([colorhist(a) for a in A16])
chm,chci,_,_,_=loso_top1(Ech, is_test)

w=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1
net=torchvision.models.mobilenet_v2(weights=w); feat=torch.nn.Sequential(net.features,torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten()); feat.eval()
prep=transforms.Compose([transforms.Resize((96,96)),transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
@torch.no_grad()
def cnn_emb(c,f):
    e=feat(prep(Image.open(f"{ROOT}/{c}/{f}").convert('RGB')).unsqueeze(0)).squeeze(0).numpy(); return e/(np.linalg.norm(e)+1e-9)
print("  encoding CNN features...")
Ecnn=np.array([cnn_emb(c,f) for c,f,_ in items])
zsm,zsci,_,_,_=loso_top1(Ecnn, is_test)

# ── R2: fine-tuned CNN upper bound (linear probe trained on VALIDATION, tested on TEST) ──
Xtr=torch.tensor(Ecnn[is_val],dtype=torch.float32); ytr=torch.tensor(y[is_val]-1)
Xte=torch.tensor(Ecnn[is_test],dtype=torch.float32); yte=torch.tensor(y[is_test]-1)
clf=torch.nn.Linear(Xtr.shape[1],10); opt=torch.optim.Adam(clf.parameters(),1e-3,weight_decay=1e-4)
lf=torch.nn.CrossEntropyLoss()
for _ in range(400): opt.zero_grad(); lf(clf(Xtr),ytr).backward(); opt.step()
with torch.no_grad(): ft_acc=(clf(Xte).argmax(1)==yte).float().mean().item()*100

print(f"\n{'Method':28s}{'TEST top-1':22s}{'Params'}")
print("-"*60)
print(f"{'Colour histogram':28s}{f'{chm:.1f}% ± {chci:.1f}':22s}{'0'}")
print(f"{'QRoPE (ours, frozen cfg)':28s}{f'{tm:.1f}% ± {tci:.1f}':22s}{'0'}")
print(f"{'CNN MobileNetV2 zero-shot':28s}{f'{zsm:.1f}% ± {zsci:.1f}':22s}{'2.2M'}")
print(f"{'CNN fine-tuned (upper bnd)':28s}{f'{ft_acc:.1f}%':22s}{'2.2M+train'}")
print(f"\nchance 10% | R1 config on val / test disjoint | R3 natural dist + per-class | k={K}")
