import os
import lmdb
from tqdm import tqdm
from PIL import Image
import numpy as np
import io
import json # To store metadata like shape

def create_image_lmdb(
    image_dir: str,
    lmdb_path: str,
    img_size: tuple = (256, 256),  # Changed default to 256x256
    store_as_encoded: bool = False,
    quality: int = 95
):
    """
    Converts images from a directory into an LMDB dataset.

    Args:
        image_dir (str): Path to the directory containing input images.
        lmdb_path (str): Path where the LMDB dataset will be created.
        img_size (tuple): Target size for resizing images (width, height).
                          Images will be resized to this resolution.
        store_as_encoded (bool): If True, images are stored as compressed JPEG bytes.
                                 If False, images are stored as raw NumPy array bytes (uint8).
        quality (int): JPEG compression quality (1-100) if store_as_encoded is True.
    """
    os.makedirs(os.path.dirname(lmdb_path), exist_ok=True)

    # Get all image files
    img_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))])
    if not img_files:
        print(f"No image files found in {image_dir}. Exiting.")
        return

  
    map_size = 1099511627776 # 1 TB 

    env = lmdb.open(lmdb_path, map_size=map_size)
    txn = env.begin(write=True)

    print(f"Starting to build LMDB at {lmdb_path}...")
    print(f"Target image size: {img_size}")
    print(f"Storing as {'encoded (JPEG)' if store_as_encoded else 'raw pixel data (NumPy)'}")

    num_processed_images = 0
    for idx, fname in enumerate(tqdm(img_files, desc="Processing images")):
        path = os.path.join(image_dir, fname)
        try:
            with Image.open(path) as pil_img:
                # Resize and convert to RGB (standardize channel order)
                pil_img = pil_img.resize(img_size).convert("RGB")

                if store_as_encoded:
                    # Store as compressed JPEG bytes
                    img_byte_arr = io.BytesIO()
                    pil_img.save(img_byte_arr, format='jpeg', quality=quality)
                    img_bytes = img_byte_arr.getvalue()
                else:
                    # Store as raw NumPy array bytes (HWC, uint8)
                    img_array = np.asarray(pil_img, dtype=np.uint8)
                    img_bytes = img_array.tobytes()

            key = str(idx).encode('ascii') # Store keys as '0', '1', '2', ...
            txn.put(key, img_bytes)
            num_processed_images += 1

            # Commit periodically to prevent too much memory usage
            if (idx + 1) % 1000 == 0:
                txn.commit()
                txn = env.begin(write=True)

        except Exception as e:
            print(f"Skipping {fname} due to error: {e}")

    # Final commit for any remaining entries
    txn.commit()

    # Store metadata: total number of images and image shape
    with env.begin(write=True) as txn:
        txn.put(b'__len__', str(num_processed_images).encode('ascii'))
        if not store_as_encoded:
            # For raw pixel data, shape (H, W, C) is crucial for reshaping
            metadata = {'shape': (img_size[1], img_size[0], 3), 'dtype': 'uint8'}
        else:
            # For encoded data, shape isn't fixed until decoded, but we can store the target size
            metadata = {'target_size': img_size, 'format': 'jpeg'}
        txn.put(b'__metadata__', json.dumps(metadata).encode('ascii'))

    env.close()
    print(f"\nLMDB dataset created at {lmdb_path} with {num_processed_images} images.")
    print(f"Stored as {'encoded' if store_as_encoded else 'raw pixel data'}.")


if __name__ == "__main__":
    source_image_folder = "./real_samples" # <-- CHANGE THIS
    output_lmdb_path = "./train.lmdb" # <-- CHANGE THIS


    target_image_resolution = (256, 256) # Default to 256x256


    store_images_as_encoded = False

    create_image_lmdb(
        image_dir=source_image_folder,
        lmdb_path=output_lmdb_path,
        img_size=target_image_resolution,
        store_as_encoded=store_images_as_encoded
    )

    print("\nLMDB creation script finished. You can now use this LMDB for training.")
    print(f"Remember to adjust your dataset loading code (MyLMDBDataset) based on `store_images_as_encoded` setting.")

