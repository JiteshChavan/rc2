import os, io, glob, json, lmdb
from tqdm import tqdm

def create_image_lmdb(image_dir, lmdb_path, commit_interval=2000, key_width=6):
    os.makedirs(os.path.dirname(lmdb_path) or ".", exist_ok=True)

    exts = ("*.png","*.jpg","*.jpeg","*.webp")
    img_files = []
    for pat in exts:
        img_files += glob.glob(os.path.join(image_dir, "**", pat), recursive=True)
    img_files.sort()
    if not img_files:
        print(f"No images under {image_dir}"); return

    total_bytes = 0
    for p in img_files:
        try: total_bytes += os.path.getsize(p)
        except OSError: pass
    # FFHQ ~90GB PNG - 1.35× headroom
    map_size = max(int(total_bytes * 1.35), 1 << 30)

    env = lmdb.open(
        lmdb_path,
        map_size=map_size,
        subdir=True,      
        lock=True,
        readahead=False,  # faster for bulk write
        meminit=False,
        map_async=True,   # async flush
        writemap=False,   
        max_dbs=1,
    )

    print(f"Building LMDB at {lmdb_path} from {len(img_files)} images")
    txn = env.begin(write=True)

    kept = 0
    key_lines = []
    for idx, p in enumerate(tqdm(img_files, desc="Writing")):
        try:
            with open(p, "rb") as f:
                data = f.read()
            key = str(idx).encode("ascii")
            txn.put(key, data)
            key_lines.append(f"{idx}\t{os.path.basename(p)}")
            kept += 1
            if (idx + 1) % commit_interval == 0:
                txn.commit(); txn = env.begin(write=True)
        except Exception as e:
            print(f"Skip {os.path.basename(p)}: {e}")

    txn.commit()

    meta = {
        "encoded": True,
        "extensions": list(exts),
        "nominal_shape": [1024, 1024, 3],  
        "count": kept,
    }
    with env.begin(write=True) as mtxn:
        mtxn.put(b"__len__", str(kept).encode("ascii"))
        mtxn.put(b"__metadata__", json.dumps(meta).encode("utf-8"))
        mtxn.put(b"__keys__", ("\n".join(key_lines)).encode("utf-8"))

    env.sync(); env.close()
    print(f"Done. {kept} images → {lmdb_path}")

if __name__ == "__main__":
    image_dir = "real_samples"  
    out_path  = "train.lmdb"
    create_image_lmdb(image_dir, out_path)
