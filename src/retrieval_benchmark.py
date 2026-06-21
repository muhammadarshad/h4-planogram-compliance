"""
Rigorous product-identification benchmark (leave-one-shelf-out).
Mine a catalog of product encodings; for each held-out shelf, identify its
products by k-NN retrieval against all OTHER shelves' crops. Report top-1
identification accuracy aggregated across shelves (mean +/- 95% CI), for
QRoPE (parameter-free) vs colour-histogram vs CNN (MobileNetV2 zero-shot).

Protocol:
  - classes 1..10 (class 0 = background, excluded — not a product to identify)
  - leave-one-SHELF-out: query shelf's crops never appear in its own gallery
  - per-class cap for balance; k=11 NN, majority vote; metric = top-1 accuracy
"""
import os, re, random, collections, math
import numpy as np
from PIL import Image
import torch, torchvision
from torchvision import transforms

ROOT='grocerydataset/ProductImagesFromShelves'; CLASSES=[str(c) for c in range(1,11)]
CAP=150; K=11; rng=random.Random(0)
src=lambda fn:(re.match(r'(.+\.JPG)_',fn).group(1) if re.match(r'(.+\.JPG)_',fn) else fn)

# gather a balanced crop set
items=[]
for c in CLASSES:
    fs=os.listdir(f"{ROOT}/{c}"); rng.shuffle(fs)
    for f in fs[:CAP]: items.append((c,f,src(f)))
y=np.array([int(c) for c,_,_ in items]); grp=[s for _,_,s in items]
print(f"catalog: {len(items)} crops, {len(CLASSES)} classes, {len(set(grp))} source shelves")

# ── encoders ──
def load(c,f,sz): return np.asarray(Image.open(f"{ROOT}/{c}/{f}").convert('RGB').resize((sz,sz)),np.float64)
def enc_qrope(arr,G=16,FREQS=range(1,9)):
    im=arr-128; rr,cc=np.mgrid[0:G,0:G]; rr,cc=rr.ravel(),cc.ravel()
    x0,x1,x2,x3=im[:,:,0].ravel(),im[:,:,1].ravel(),im[:,:,2].ravel(),im.mean(2).ravel()
    sig=[]
    for fr in FREQS:
        for fc in FREQS:
            t1=2*np.pi*fr*rr/G; t2=2*np.pi*fc*cc/G
            c1,s1,c2,s2=np.cos(t1),np.sin(t1),np.cos(t2),np.sin(t2)
            sig+=[(x0*c1-x1*s1).sum(),(x0*s1+x1*c1).sum(),(x2*c2-x3*s2).sum(),(x2*s2+x3*c2).sum()]
    s=np.array(sig); return s/(np.linalg.norm(s)+1e-9)
def enc_colorhist(arr):
    q=(arr.reshape(-1,3)//64).astype(int); idx=q[:,0]*16+q[:,1]*4+q[:,2]
    h=np.bincount(idx,minlength=64).astype(float); return h/(np.linalg.norm(h)+1e-9)

# CNN embeddings
w=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1
net=torchvision.models.mobilenet_v2(weights=w); net.classifier=torch.nn.Identity(); net.eval()
prep=transforms.Compose([transforms.Resize((96,96)),transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
@torch.no_grad()
def enc_cnn(c,f):
    e=net(prep(Image.open(f"{ROOT}/{c}/{f}").convert('RGB')).unsqueeze(0)).squeeze(0).numpy()
    return e/(np.linalg.norm(e)+1e-9)

print("encoding (qrope, colour-hist, cnn)...")
A16=[load(c,f,16) for c,f,_ in items]
Eq=np.array([enc_qrope(a) for a in A16])
Ec=np.array([enc_colorhist(a) for a in A16])
En=np.array([enc_cnn(c,f) for c,f,_ in items])

# ── leave-one-shelf-out top-1 identification ──
def loso_accuracy(E):
    by_shelf=collections.defaultdict(list)
    for i,s in enumerate(grp): by_shelf[s].append(i)
    accs=[]
    for s,qi in by_shelf.items():
        gal=[i for i in range(len(items)) if grp[i]!=s]
        if len(qi)<3: continue                      # need a few products to score a shelf
        Gv=E[gal]; gyy=y[gal]
        sims=E[qi]@Gv.T
        correct=0
        for r,i in enumerate(qi):
            idx=np.argsort(-sims[r])[:K]
            pred=collections.Counter(int(gyy[j]) for j in idx).most_common(1)[0][0]
            correct+=(pred==int(y[i]))
        accs.append(correct/len(qi))
    accs=np.array(accs)
    ci=1.96*accs.std(ddof=1)/math.sqrt(len(accs))
    return accs.mean()*100, ci*100, len(accs)

print(f"\n{'Method':22s}{'top-1 identification (mean ± 95% CI)':32s}{'params'}")
print("-"*64)
for name,E,p in [("QRoPE (ours)",Eq,"0"),("Colour histogram",Ec,"0"),("CNN MobileNetV2 z.s.",En,"2.2M")]:
    m,ci,n=loso_accuracy(E)
    print(f"{name:22s}{f'{m:.1f}% ± {ci:.1f}  (n={n} shelves)':32s}{p}")
print(f"\nchance = {100/len(CLASSES):.0f}%   | leave-one-shelf-out, k={K} NN, classes 1-10")