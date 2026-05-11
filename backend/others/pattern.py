#!/usr/bin/env python3
"""
Lightweight defect pattern detector (no torch required).
- Reads /mnt/data/session_20260223_135818.csv
- Uses raw_image + yolo_bboxes (if present) to crop defect
- Builds simple embeddings: resize->grayscale->flatten -> PCA
- Clusters embeddings with DBSCAN and reports clusters that look suspicious
"""
import os, ast
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from collections import Counter
from datetime import timedelta
BASE_DIR = "/home/aswin/Documents/sam3_impl"
CSV_PATH = "logs/no_issue_session.csv"
OUT_CSV = "pattern_clusters_fallback.csv"
EXAMPLE_DIR = "cluster_examples_fallback"
os.makedirs(EXAMPLE_DIR, exist_ok=True)

# Config (tune these for your small dataset)
IMG_SIZE = (128, 128)    # crop -> resized to this
PCA_DIM = 64
EPS = 0.30               # DBSCAN eps on cosine (tweak: 0.25..0.45)
MIN_SAMPLES = 3          # smallest cluster size (use 3 for 58 samples)
MIN_CLUSTER_SIZE_ALERT = 3
PURITY_THRESHOLD = 0.6
TIME_WINDOW_MINUTES = 30

df = pd.read_csv(CSV_PATH)
print("Columns:", df.columns.tolist())
df['image_path'] = df['raw_image']
df['label'] = df.get('yolo_labels', None)
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

def parse_bboxes(cell):
    if pd.isna(cell):
        return []
    s = str(cell).strip()
    try:
        val = ast.literal_eval(s)
        if isinstance(val, (list,tuple)):
            if len(val) == 0:
                return []
            # single bbox [x1,y1,x2,y2] or list of bboxes
            if all(isinstance(x,(int,float)) for x in val) and len(val)==4:
                return [tuple(val)]
            out=[]
            for item in val:
                if isinstance(item,(list,tuple)) and len(item)==4:
                    out.append(tuple(item))
            return out
    except Exception:
        pass
    # fallback parse
    parts = [p for p in s.replace('|',';').split(';') if p.strip()]
    out=[]
    for p in parts:
        nums = [t for t in p.replace(',',' ').split() if t.strip()]
        if len(nums)>=4:
            try:
                out.append(tuple(map(float, nums[:4])))
            except:
                pass
    return out

df['bboxes'] = df.get('yolo_bboxes', '').apply(parse_bboxes)

# resolve image path helper: try /mnt/data and absolute path
def resolve(p):
    if not isinstance(p, str):
        return None

    # 1. Absolute path
    if os.path.isabs(p) and os.path.exists(p):
        return p

    # 2. Relative to project base
    candidate = os.path.join(BASE_DIR, p)
    if os.path.exists(candidate):
        return candidate

    # 3. Search by filename inside BASE_DIR
    fname = os.path.basename(p)
    for root, _, files in os.walk(BASE_DIR):
        if fname in files:
            return os.path.join(root, fname)

    return None

df['resolved'] = df['image_path'].apply(resolve)
df = df.dropna(subset=['resolved']).reset_index(drop=True)
print("Valid entries:", len(df))

def load_crop(path, bboxes):
    try:
        im = Image.open(path).convert("RGB")
    except:
        return None
    w,h = im.size
    crop = None
    if bboxes:
        # choose largest bbox
        b = max(bboxes, key=lambda x: (x[2]-x[0])*(x[3]-x[1]))
        x1,y1,x2,y2 = b
        # if normalized coords, convert
        if 0<=x1<=1 and 0<=y1<=1 and 0<=x2<=1 and 0<=y2<=1:
            x1*=w; x2*=w; y1*=h; y2*=h
        x1,x2 = int(max(0,min(w,x1))), int(max(0,min(w,x2)))
        y1,y2 = int(max(0,min(h,y1))), int(max(0,min(h,y2)))
        if x2-x1>8 and y2-y1>8:
            crop = im.crop((x1,y1,x2,y2))
    if crop is None:
        # fallback center crop
        crop = ImageOps.fit(im, (int(w*0.9), int(h*0.9)), method=Image.BICUBIC)
    crop = crop.resize(IMG_SIZE).convert("L")  # grayscale
    return np.array(crop).astype(np.float32).ravel() / 255.0

# build embeddings
rows=[]
X=[]
for i,row in df.iterrows():
    p = row['resolved']
    b = row['bboxes']
    v = load_crop(p,b)
    if v is None:
        continue
    rows.append({
        "index": i,
        "path": p,
        "label": row.get('label', None),
        "timestamp": row.get('timestamp', pd.NaT)
    })
    X.append(v)
if len(X)==0:
    raise SystemExit("No images loaded.")

X = np.vstack(X)
print("Raw vectors shape:", X.shape)

# PCA reduce
pca = PCA(n_components=min(PCA_DIM, X.shape[0]-1))
Z = pca.fit_transform(X)
# normalize rows
Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)

# DBSCAN on cosine (cosine distance = 1 - dot)
db = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES, metric='cosine')
labels = db.fit_predict(Z)
print("Clusters (label counts):", Counter(labels))

# build output table
out = []
for i, r in enumerate(rows):
    out.append({
        "index": r['index'],
        "path": r['path'],
        "label": r['label'],
        "timestamp": r['timestamp'],
        "cluster": int(labels[i])
    })
out_df = pd.DataFrame(out)
out_df.to_csv(OUT_CSV, index=False)
print("Wrote cluster assignments to", OUT_CSV)

# analyze clusters for alerts
alerts=[]
for cid in sorted(out_df['cluster'].unique()):
    if cid == -1: continue
    grp = out_df[out_df['cluster']==cid]
    size = len(grp)
    purity = 0.0
    top_label=None
    if size>0:
        lc = Counter(grp['label'].fillna("NA").tolist())
        top_label, top_count = lc.most_common(1)[0]
        purity = top_count/size
    times = pd.to_datetime(grp['timestamp'], errors='coerce').dropna()
    time_span = None
    if len(times)>0:
        time_span = (times.max()-times.min()).total_seconds()/60.0
    print(f"Cluster {cid}: size={size}, purity={purity:.2f}, top_label={top_label}, time_span_min={time_span}")
    if size>=MIN_CLUSTER_SIZE_ALERT and purity>=PURITY_THRESHOLD and (time_span is None or time_span<=TIME_WINDOW_MINUTES):
        alerts.append((cid,size,top_label,purity,time_span))
    # save a few sample images for quick inspection
    for j,p_row in enumerate(grp['path'].tolist()[:4]):
        try:
            im = Image.open(p_row).convert("RGB")
            im.resize((512,512)).save(os.path.join(EXAMPLE_DIR, f"cluster{cid}_{j}.jpg"))
        except:
            pass

if alerts:
    print("\n=== MACHINE-LEVEL ALERTS ===")
    for a in alerts:
        print("Cluster", a)
else:
    print("\nNo alerts by fallback rules (lowered thresholds). Check cluster_examples at", EXAMPLE_DIR)

print("Done.")