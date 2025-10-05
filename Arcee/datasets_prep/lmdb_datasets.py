import io
import os
import lmdb
import numpy as np
import torch.utils.data as data
from PIL import Image
import json # Import json to read metadata

# The num_samples function is likely used elsewhere, keep it as is
def num_samples(dataset, train):
    if dataset == "celeba":
        return 27000 if train else 3000
    elif dataset == "lsun_church":
        return 126227
    else:
        raise NotImplementedError("dataset %s is unknown" % dataset)


class LMDBDataset(data.Dataset):
    def __init__(self, root, name="", train=True, transform=None, is_encoded=False):
        self.train = train
        self.name = name
        self.transform = transform
        
        # Determine LMDB path based on train/validation
        if self.train:
            lmdb_path = os.path.join(root, "train.lmdb")
        else:
            lmdb_path = os.path.join(root, "validation.lmdb")
        self.lmdb_path = lmdb_path
        
        self.is_encoded = is_encoded
        self.env = None # Initialize env to None, will be opened in open_lmdb

        # Open LMDB environment once during initialization
        self.open_lmdb() 
        
        # Get the size using the __len__ key from metadata
        with self.env.begin() as txn:
            len_bytes = txn.get(b'__len__')
            if len_bytes:
                self.size = int(len_bytes.decode('ascii'))
                print(f"DEBUG(LMDBDataset): Initialized with size from '__len__' key: {self.size}")
            else:
                # Fallback if __len__ key is missing (though your LMDB has it)
                # This will count all entries including metadata, so it's less precise
                self.size = txn.stat()["entries"] - 2 # Subtract 2 for __len__ and __metadata__ keys
                print(f"DEBUG(LMDBDataset): Initialized with size from stat() - 2: {self.size}")
                print("DEBUG(LMDBDataset): WARNING: '__len__' key not found or invalid. Using txn.stat()['entries'] - 2.")

            # Load metadata for image shape if raw pixel data
            if not self.is_encoded:
                metadata_bytes = txn.get(b'__metadata__')
                if metadata_bytes:
                    metadata = json.loads(metadata_bytes.decode('ascii'))
                    if 'shape' in metadata:
                        self.img_shape = tuple(metadata['shape'])
                        print(f"DEBUG(LMDBDataset): Loaded image shape from metadata: {self.img_shape}")
                    else:
                        # Fallback if shape not in metadata, assume a default (e.g., 256x256x3)
                        print("DEBUG(LMDBDataset): WARNING: 'shape' not in metadata. Assuming 256x256x3.")
                        self.img_shape = (256, 256, 3) # Adjust this if your images are a different default size
                else:
                    print("DEBUG(LMDBDataset): WARNING: No '__metadata__' key found. Assuming 256x256x3.")
                    self.img_shape = (256, 256, 3) # Adjust this if your images are a different default size

        # DO NOT close self.env here. It must remain open for __getitem__.
        # self.env.close() # REMOVE THIS LINE
        # del self.env # REMOVE THIS LINE

    def open_lmdb(self):
        """Opens the LMDB environment if it's not already open."""
        if self.env is None: # Only open if not already open
            self.env = lmdb.open(self.lmdb_path, readonly=True, max_readers=1,
                                 lock=False, readahead=False, meminit=False)
            print(f"DEBUG(LMDBDataset): LMDB environment opened for {self.lmdb_path}")

    def __getitem__(self, index):
        # No need for `if not hasattr(self, "env"): self.open_lmdb()` here
        # because self.env is guaranteed to be open from __init__
        
        # Add this print to track requested indices
        # print(f"DEBUG(LMDBDataset): __getitem__ requesting index: {index}")

        with self.env.begin(write=False, buffers=True) as txn:
            data = txn.get(str(index).encode())
            
            # Crucial check: if data is None, the key doesn't exist
            if data is None:
                print(f"ERROR(LMDBDataset): Key '{index}' returned None. This indicates a mismatch between __len__ and actual keys.")
                # You can raise an IndexError or handle it as appropriate for your DataLoader
                raise IndexError(f"LMDB key '{index}' not found. Dataset length might be incorrect or keys are missing.")

            if self.is_encoded:
                img = Image.open(io.BytesIO(data))
                img = img.convert("RGB")
            else:
                img = np.asarray(data, dtype=np.uint8)
                # Use the stored img_shape for reshaping
                img = np.reshape(img, self.img_shape)
                img = Image.fromarray(img, mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        target = [0] # Assuming target is always [0] as per your original code
        return img, target

    def __len__(self):
        return self.size

# You will need to ensure your main training script (train.py)
# uses this modified LMDBDataset class definition.
# Also, ensure your LMDB creation script (create_image_lmdb)
# correctly stores the '__len__' and '__metadata__' keys.