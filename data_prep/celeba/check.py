import os
import lmdb
import io
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as data # Assuming this is a PyTorch-style Dataset

class MyLMDBDataset(data.Dataset):
    """
    A simplified Dataset class to read images from an LMDB database,
    using the __getitem__ logic you provided.
    """
    def __init__(self, lmdb_path, is_encoded=False, transform=None):
        self.lmdb_path = lmdb_path
        self.is_encoded = is_encoded # This is the crucial flag!
        self.transform = transform
        self.env = None # Initialize env as None

        # Open LMDB environment once when the dataset object is created
        self.open_lmdb()
        
        # Now get the length using the already opened environment
        self.length = self._get_db_length() 

    def open_lmdb(self):
        """Opens the LMDB environment if it's not already open."""
        if self.env is None:
            self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)

    def _get_db_length(self):
        """
        Estimates the number of entries in the LMDB using the already opened environment (self.env).
        Assumes keys are 0-indexed integers.
        """
        if self.env is None:
            raise RuntimeError("LMDB environment must be open before calling _get_db_length.")

        with self.env.begin(write=False) as txn:
            # Check for a special length key first (common practice for large LMDBs)
            length_bytes = txn.get(b'__len__')
            if length_bytes:
                return int(length_bytes.decode('ascii'))
            else:
                # Fallback: Count entries by iterating (can be slow for very large LMDBs)
                print("Warning: '__len__' key not found. Counting LMDB entries (may be slow).")
                count = 0
                cursor = txn.cursor()
                for key, _ in cursor:
                    try:
                        # Attempt to decode as integer key, skip metadata keys
                        int(key.decode('ascii'))
                        count += 1
                    except ValueError:
                        pass # Skip non-integer keys (like '__len__', if it existed)
                return count

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # No need to check self.env here, as it's opened in __init__
        # and remains open.
        
        with self.env.begin(write=False, buffers=True) as txn:
            data = txn.get(str(index).encode())
            if data is None:
                raise IndexError(f"Index {index} not found in LMDB.")

            if self.is_encoded:
                img = Image.open(io.BytesIO(data))
                img = img.convert("RGB")
            else:
                img = np.asarray(data, dtype=np.uint8)
                # Calculate size based on the total number of bytes and 3 channels (RGB)
                # Your LMDB creation uses (68, 68, 3)
                size = int(np.sqrt(len(img) / 3))
                img = np.reshape(img, (size, size, 3))
                img = Image.fromarray(img, mode="RGB")

        if self.transform:
            img = self.transform(img)

        return img # Return just the image for plotting purposes

def plot_images_from_dataset(dataset, num_images=5):
    """
    Plots a specified number of images from a dataset.
    """
    print(f"Attempting to plot {num_images} images from the dataset.")
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 4, 4))
    if num_images == 1:
        axes = [axes] # Ensure axes is iterable even for 1 image

    images_plotted = 0
    # Use range(len(dataset)) to ensure we don't go out of bounds
    for i in range(min(num_images, len(dataset))):
        try:
            img = dataset[i] # Get image using __getitem__
            axes[images_plotted].imshow(img)
            axes[images_plotted].set_title(f"Image {i}")
            axes[images_plotted].axis('off')
            images_plotted += 1
        except Exception as e:
            print(f"Could not retrieve/plot image at index {i}: {e}")
            # If an error occurs, try to plot fewer images if possible
            if images_plotted == 0:
                print("No images could be plotted successfully.")
                plt.close(fig) # Close empty figure
                return

    if images_plotted > 0:
        plt.tight_layout()
        plt.savefig("lmdb_dataset_sample_images.png")
        print(f"Saved {images_plotted} sample images to lmdb_dataset_sample_images.png")
    else:
        print("No images were successfully plotted.")
        plt.close(fig) # Close empty figure


if __name__ == "__main__":
    lmdb_shard_path = "./train.lmdb" # Your LMDB path

    # Instantiate the dataset.
    # IMPORTANT: Set is_encoded=False because you stored raw NumPy bytes.
    dataset = MyLMDBDataset(lmdb_path=lmdb_shard_path, is_encoded=False)

    print(f"Dataset initialized. Found {len(dataset)} entries in LMDB.")

    # Plot images from the dataset
    plot_images_from_dataset(dataset, num_images=5)

    # It's crucial to close the LMDB environment when done
    if dataset.env:
        dataset.env.close()
        print("LMDB environment closed.")