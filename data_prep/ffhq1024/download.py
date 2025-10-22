import os, glob, tarfile
from huggingface_hub import snapshot_download

path = snapshot_download(repo_id="gaunernst/ffhq-1024-wds", repo_type="dataset", local_dir="_ffhq_hf")
out = "real_samples"
os.makedirs(out, exist_ok=True)
count = 0
for t in sorted(glob.glob(os.path.join(path, "*.tar"))):
    with tarfile.open(t) as tf:
        for m in tf.getmembers():
            if not m.name.endswith(".webp"): continue
            f = tf.extractfile(m)
            if f is None: continue
            with open(os.path.join(out, os.path.basename(m.name)), "wb") as w:
                w.write(f.read())
            count += 1
print("Extracted", count, "WEBP files to ./real_samples")
