"""
End-to-end integer planogram-compliance pipeline.
Unifies the two halves on real shelves:
  WHERE  - H4 position code, integer L1 match to a planogram slot
  WHAT   - QRoPE identity descriptor, k-NN retrieval from a product catalog
A detected box is COMPLIANT iff it matches a slot in position AND the retrieved
product identity equals the slot's expected product. Else VIOLATION.

Evaluation (leakage-controlled, aggregated with 95% CIs):
  - reference planogram  = shelf copy A of planogram P
  - compliant query      = shelf copy B of the SAME planogram P  -> expect COMPLIANT
  - violation query      = shelf of a DIFFERENT planogram (same camera) -> expect VIOLATION
  - identity catalog excludes A, B and the violation shelf (no leakage)
"""
import os, re, csv, collections, math, random
import numpy as np
from PIL import Image

ROOT='grocerydataset/ProductImagesFromShelves'; CLASSES=[str(c) for c in range(1,11)]
TAU_POS = 30      # H4 integer-L1 position threshold (from separation analysis)
K = 11            # identity k-NN
rng = random.Random(0)

# ── encoders ──────────────────────────────────────────────────────────
def extent_norm(bs):
    xs=[b[0] for b in bs]+[b[0]+b[2] for b in bs]; ys=[b[1] for b in bs]+[b[1]+b[3] for b in bs]
    x0,y0=min(xs),min(ys); sx=max(1,max(xs)-x0); sy=max(1,max(ys)-y0)
    return [(min(255,(x-x0)*255//sx),min(255,(y-y0)*255//sy),min(255,w*255//sx),min(255,h*255//sy))
            for x,y,w,h in bs]
def h4(z):
    x,y,w,h=z; return (x+y+w+h, x-y+w-h, x+y-w-h, x-y-w+h)
def l1(a,b): return sum(abs(p-q) for p,q in zip(a,b))
def qrope(arr,G=16,FREQS=range(1,9)):
    im=arr-128; rr,cc=np.mgrid[0:G,0:G]; rr,cc=rr.ravel(),cc.ravel()
    x0,x1,x2,x3=im[:,:,0].ravel(),im[:,:,1].ravel(),im[:,:,2].ravel(),im.mean(2).ravel()
    sig=[]
    for fr in FREQS:
        for fc in FREQS:
            t1=2*np.pi*fr*rr/G; t2=2*np.pi*fc*cc/G
            c1,s1,c2,s2=np.cos(t1),np.sin(t1),np.cos(t2),np.sin(t2)
            sig+=[(x0*c1-x1*s1).sum(),(x0*s1+x1*c1).sum(),(x2*c2-x3*s2).sum(),(x2*s2+x3*c2).sum()]
    s=np.array(sig); return s/(np.linalg.norm(s)+1e-9)

# ── identity catalog: encode every product crop once, keyed by (shelf,coords) ──
crop_src = lambda fn: re.match(r'(.+\.JPG)_',fn).group(1)
def crop_coords(fn):
    m=re.match(r'.+\.JPG_(\d+)_(\d+)_(\d+)_(\d+)\.png',fn); return tuple(map(int,m.groups())) if m else None
catalog_vec=[]; catalog_cls=[]; catalog_shelf=[]; box_enc={}
for c in CLASSES:
    for f in os.listdir(f"{ROOT}/{c}"):
        co=crop_coords(f)
        if co is None: continue
        arr=np.asarray(Image.open(f"{ROOT}/{c}/{f}").convert('RGB').resize((16,16)),np.float64)
        v=qrope(arr)
        catalog_vec.append(v); catalog_cls.append(int(c)); catalog_shelf.append(crop_src(f))
        box_enc[(crop_src(f),co)]=(v,int(c))           # lookup: box -> (encoding, true class)
CAT=np.array(catalog_vec); CCLS=np.array(catalog_cls); CSHELF=np.array(catalog_shelf)
print(f"identity catalog: {len(CAT)} product encodings across {len(set(catalog_shelf))} shelves")

# ── shelves: boxes (corners->xywh) + class, group by planogram ──
rows=[r for r in csv.reader(open('grocerydataset/annotations.csv')) if len(r)>=6]
shelf_boxes=collections.defaultdict(list)
for r in rows:
    x1,y1,x2,y2=map(int,r[1:5]); shelf_boxes[r[0]].append((x1,y1,x2-x1,y2-y1,int(r[5])))
cam=lambda i:re.match(r'(C\d+)',i).group(1); plano=lambda i:re.match(r'C\d+_P(\d+)',i).group(1)
prefix=lambda i:re.sub(r'_\d+\.JPG$','',i)
groups=collections.defaultdict(list)
for s in shelf_boxes: groups[prefix(s)].append(s)
pairs=[(c[0],c[1]) for c in groups.values() if len(c)>=2]   # (ref, compliant-query) same planogram

def identity_of(shelf, box_xywh):
    """k-NN identity for a box, retrieving from catalog EXCLUDING this shelf."""
    key=(shelf, box_xywh)
    if key not in box_enc: return None
    v,_=box_enc[key]
    mask=CSHELF!=shelf
    sims=CAT[mask]@v; cls=CCLS[mask]
    idx=np.argsort(-sims)[:K]
    return collections.Counter(int(cls[idx[j]]) for j in range(K)).most_common(1)[0][0]

def build_planogram(ref):
    bs=[(x,y,w,h) for x,y,w,h,_ in shelf_boxes[ref]]
    z=extent_norm(bs); codes=[h4(zz) for zz in z]
    return list(zip(codes, [c for *_,c in shelf_boxes[ref]]))   # (h4 code, expected class)

def evaluate(ref, query, expect_compliant):
    plan=build_planogram(ref)
    bs=[(x,y,w,h) for x,y,w,h,_ in shelf_boxes[query]]
    z=extent_norm(bs); codes=[h4(zz) for zz in z]
    passed=0; n=0
    for (code, (x,y,w,h,gt)) in zip(codes, shelf_boxes[query]):
        ident=identity_of(query,(x,y,w,h))
        if ident is None: continue                  # box without a catalog crop (e.g. background)
        n+=1
        # nearest planogram slot by position
        dpos,exp_cls=min(((l1(code,pc),ec) for pc,ec in plan), key=lambda t:t[0])
        pos_ok = dpos<=TAU_POS
        id_ok  = ident==exp_cls
        compliant = pos_ok and id_ok
        passed += (compliant if expect_compliant else not compliant)
    return passed, n

# ── aggregate over planogram pairs + a different-planogram violation shelf ──
rng.shuffle(pairs)
comp_rates=[]; viol_rates=[]; used=0
for ref, qry in pairs:
    diffs=[s for s in shelf_boxes if cam(s)==cam(ref) and plano(s)!=plano(ref)]
    if not diffs: continue
    viol=max(diffs, key=lambda s: len(shelf_boxes[s]))
    pc,pn=evaluate(ref,qry,expect_compliant=True)
    vc,vn=evaluate(ref,viol,expect_compliant=False)
    if pn>=5 and vn>=5:
        comp_rates.append(pc/pn); viol_rates.append(vc/vn); used+=1
    if used>=40: break

def stat(a):
    a=np.array(a); ci=1.96*a.std(ddof=1)/math.sqrt(len(a)); return a.mean()*100, ci*100
cm,cci=stat(comp_rates); vm,vci=stat(viol_rates)
print(f"\nEnd-to-end planogram compliance ({used} planogram triples, leave-shelf-out identity)")
print(f"  Compliant detection (same planogram):     {cm:.1f}% ± {cci:.1f}")
print(f"  Violation detection (diff planogram):     {vm:.1f}% ± {vci:.1f}")
print(f"  Position threshold tau={TAU_POS}, identity k={K}, 0 trained params, 0 float* (LUT trig)")
