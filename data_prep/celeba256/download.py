from datasets import load_dataset
import os

ds = load_dataset("korexyz/celeba-hq-256x256", split="train")

os.makedirs("real_samples", exist_ok=True)

for i, ex in enumerate(ds):
    img = ex["image"]  # Already a PIL Image
    img.save(f"real_samples/{i:05d}.jpg")
