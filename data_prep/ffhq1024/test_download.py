import os, tarfile
from huggingface_hub import hf_hub_download

REPO = "gaunernst/ffhq-1024-wds"
SHARD = 0                     
FNAME = f"{SHARD:05d}.tar"    


tar_path = hf_hub_download(
    repo_id=REPO,
    repo_type="dataset",
    filename=FNAME,
    local_dir="_ffhq_hf",     
    local_dir_use_symlinks=False,
)

out = "real_samples"
os.makedirs(out, exist_ok=True)

count = 0
with tarfile.open(tar_path) as tf:
    for m in tf.getmembers():
        if not (m.isfile() and m.name.endswith(".webp")):
            continue
        src = tf.extractfile(m)
        if src is None:
            continue
        dst = os.path.join(out, os.path.basename(m.name))
        if not os.path.exists(dst):
            with open(dst, "wb") as w:
                w.write(src.read())
            count += 1

print(f"Extracted {count} WEBP files from {FNAME} into ./real_samples")