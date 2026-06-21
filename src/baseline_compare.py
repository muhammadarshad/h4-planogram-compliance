"""
Paper D — Baseline comparison.
Task: per-product planogram-compliance matching on the SAME 66 same-camera
shelf-pair benchmark used in verify.py. Every method receives the IDENTICAL
extent-normalised integer boxes — the only variable is the distance/encoding —
so the comparison isolates what the H₄ transform actually contributes vs the
standard methods a practitioner would reach for.

Metrics (per method):
  • AUC   — threshold-free power to tell a same-planogram product-match from a
            different-planogram one. 0.5 = chance, 1.0 = perfect. The headline
            quality number (no hand-tuned threshold).
  • Sep.  — separation ratio of mean diff vs mean same match score.
  • Cost  — arithmetic ops to ENCODE one box + to COMPARE two, and whether the
            pipeline needs floating point.
"""
import csv, collections, random, re
import numpy as np

ANN = 'grocerydataset/annotations.csv'

# ── Load + extent-normalise (shared front-end for all methods) ────────
rows = [r for r in csv.reader(open(ANN)) if len(r) >= 6]
px = collections.defaultdict(list)
for r in rows:
    x1, y1, x2, y2 = map(int, r[1:5])
    px[r[0]].append((x1, y1, x2 - x1, y2 - y1))

def extent_z(bs):
    xs = [x for x,y,w,h in bs] + [x+w for x,y,w,h in bs]
    ys = [y for x,y,w,h in bs] + [y+h for x,y,w,h in bs]
    x0, y0 = min(xs), min(ys); sx = (max(xs)-x0) or 1; sy = (max(ys)-y0) or 1
    return np.array([(min(255,((x-x0)*255)//sx), min(255,((y-y0)*255)//sy),
                      min(255,(w*255)//sx), min(255,(h*255)//sy)) for x,y,w,h in bs], dtype=np.int64)
Z = {i: extent_z(px[i]) for i in px}

# ── Build the benchmark pairs (identical to verify.py) ────────────────
NAME = re.compile(r'^(C\d+)_P(\d+)_N\d+_S\d+_(\d+)\.JPG$', re.I)
pre = lambda i: re.sub(r'_\d+\.JPG$','',i,flags=re.I)
cam = lambda i: NAME.match(i).group(1); plano = lambda i: NAME.match(i).group(2)
shelves = collections.defaultdict(list)
for i in px: shelves[pre(i)].append(i)
same_pairs = [(c[0],c[k]) for c in shelves.values() if len(c)>=2 for k in range(1,len(c))]
bycam = collections.defaultdict(list)
for i in px: bycam[cam(i)].append(i)
rng = random.Random(42); diff_pairs = []
for c, ims in bycam.items():
    ims = sorted(ims)
    for a in ims:
        cs = [b for b in ims if plano(b) != plano(a)]
        if cs: diff_pairs.append((a, rng.choice(cs)))
diff_pairs = rng.sample(diff_pairs, min(len(same_pairs), len(diff_pairs)))

# ── Methods: each maps (detections Z, planogram Z) -> per-detection match score.
#    Convention: HIGHER score = MORE compliant (better match), so all methods
#    point the same way for AUC.  Distances are negated.
def m_h4_l1(D, P):
    H = np.array([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]])
    Dh, Ph = D @ H.T, P @ H.T
    d = np.abs(Dh[:,None,:] - Ph[None,:,:]).sum(2)   # L1 in H₄ space
    return -d.min(1)

def m_raw_l1(D, P):                                  # ablation: no transform
    d = np.abs(D[:,None,:] - P[None,:,:]).sum(2)
    return -d.min(1)

def m_raw_l2(D, P):                                  # full 4-vec Euclidean, no transform
    d = np.sqrt(((D[:,None,:] - P[None,:,:])**2).sum(2))
    return -d.min(1)

def m_h4_l2(D, P):                                   # H₄ + full 4-vec Euclidean
    H = np.array([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]])
    Dh, Ph = D @ H.T, P @ H.T
    d = np.sqrt(((Dh[:,None,:] - Ph[None,:,:])**2).sum(2))
    return -d.min(1)

def m_center_l2(D, P):                               # box-centre Euclidean
    dc = D[:,:2] + D[:,2:]/2.0; pc = P[:,:2] + P[:,2:]/2.0
    d = np.sqrt(((dc[:,None,:] - pc[None,:,:])**2).sum(2))
    return -d.min(1)

def m_iou(D, P):                                     # field standard
    dx1,dy1 = D[:,0], D[:,1]; dx2,dy2 = D[:,0]+D[:,2], D[:,1]+D[:,3]
    px1,py1 = P[:,0], P[:,1]; px2,py2 = P[:,0]+P[:,2], P[:,1]+P[:,3]
    ix1 = np.maximum(dx1[:,None], px1[None,:]); iy1 = np.maximum(dy1[:,None], py1[None,:])
    ix2 = np.minimum(dx2[:,None], px2[None,:]); iy2 = np.minimum(dy2[:,None], py2[None,:])
    iw = np.clip(ix2-ix1, 0, None); ih = np.clip(iy2-iy1, 0, None); inter = iw*ih
    da = (D[:,2]*D[:,3])[:,None]; pa = (P[:,2]*P[:,3])[None,:]
    iou = inter / (da + pa - inter + 1e-9)
    return iou.max(1)

METHODS = [
    ("IoU (field standard)",   m_iou,       "float",   "~8 ops + 1 divide / compare",  "—"),
    ("Centre L2 (Euclidean)",  m_center_l2, "float",   "4 sub,4 mul,1 sqrt / compare", "2 add"),
    ("Raw L2 (4-vec)",         m_raw_l2,    "float",   "4 sub,4 mul,1 sqrt / compare", "0"),
    ("H₄ + L2 (4-vec)",        m_h4_l2,     "float",   "4 sub,4 mul,1 sqrt / compare", "8 int"),
    ("Raw L1 (no transform)",  m_raw_l1,    "integer", "4 sub-abs / compare",          "0 (ablation)"),
    ("H₄ + L1  (proposed)",    m_h4_l1,     "integer", "4 sub-abs / compare",          "8 int (4 add,4 sub)"),
]

def auc(pos, neg):
    """P(pos score > neg score), rank-based (Mann-Whitney)."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    allv = np.concatenate([pos, neg]); order = allv.argsort(kind='mergesort')
    ranks = np.empty_like(order, dtype=float); ranks[order] = np.arange(1, len(allv)+1)
    # average ties
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts); starts = csum - counts
    avg = (starts + csum + 1) / 2.0
    ranks = avg[inv]
    R = ranks[:len(pos)].sum()
    return (R - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg))

print(f"Benchmark: {len(same_pairs)} same-planogram pairs vs {len(diff_pairs)} different-planogram pairs (same camera)\n")
hdr = f"{'Method':24s} {'AUC':>6s} {'Sep.':>6s}  {'Arithmetic':9s} {'compare cost':30s} {'encode/box':22s}"
print(hdr); print("-"*len(hdr))
results = []
for name, fn, arith, cost, enc in METHODS:
    same = np.concatenate([fn(Z[b], Z[a]) for a,b in same_pairs])   # detections=b vs planogram=a
    diff = np.concatenate([fn(Z[b], Z[a]) for a,b in diff_pairs])
    a = auc(same, diff)
    sep = abs(diff.mean()/same.mean()) if same.mean()!=0 else float('inf')
    results.append((name, a, sep))
    print(f"{name:24s} {a:6.3f} {sep:5.2f}x  {arith:9s} {cost:30s} {enc:22s}")

best = max(results, key=lambda r: r[1])
print(f"\nHighest discrimination AUC: {best[0]} ({best[1]:.3f})")
