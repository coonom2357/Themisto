from torchvision import transforms
import numpy as np
import pandas as pd
import kagglehub
from PIL import Image
import os
from itertools import islice
from sklearn.preprocessing import MinMaxScaler
import torch
from torchvision.io import decode_image
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(
        self, annotations_file, img_dir, transform=None, target_transform=None
    ):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        
        # Create label encoding: map emotion names to integers
        self.label_to_idx = {label: idx for idx, label in enumerate(self.img_labels['label'].unique())}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = decode_image(img_path)
        # Convert image from uint8 (0-255) to float (0-1) for neural network
        image = image.float() / 255.0
        
        # Get label and convert to integer
        label_name = self.img_labels.iloc[idx, 1]
        label = self.label_to_idx[label_name]
        
        # Apply custom transforms (data augmentation) if provided
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
        

if __name__ == "__main__":
    dataset = CustomImageDataset("annotations.csv", ".")
    print(dataset[0])
    print(dataset[1])

SEARCH_ROOT_DIR = "data"

def find_jpg_files(SEARCH_ROOT_DIR, max_files=True):
    jpg_file_paths = []
    # Step 1: Find minimum number of JPG files across all subdirectories
    if max_files is True:
        max_file_counts = []
        for folder in os.listdir(SEARCH_ROOT_DIR):
            folder_path = os.path.join(SEARCH_ROOT_DIR, folder)
            if os.path.isdir(folder_path):
                count = sum(1 for f in os.listdir(folder_path) if f.lower().endswith(".jpg"))
                max_file_counts.append(count)
        max_file_limit = min(max_file_counts)
    else:
        max_file_limit = None
    
    # Step 2: Collect JPG files, limiting each subdirectory to max_file_limit
    for folder in os.listdir(SEARCH_ROOT_DIR):
        folder_path = os.path.join(SEARCH_ROOT_DIR, folder)
        if os.path.isdir(folder_path):
            count = 0 
            for file_name in os.listdir(folder_path):
                if file_name.lower().endswith(".jpg"):
                    jpg_file_paths.append(os.path.join(folder_path, file_name))
                    count += 1
                    # Stop when we reach the limit for this folder
                    if max_file_limit is not None and count >= max_file_limit:
                        break
    
    return jpg_file_paths

def generate_dataframe(jpg_file_paths):
    data = []
    for image_path in jpg_file_paths:
        label = os.path.basename(os.path.dirname(image_path))
        data.append({"image_path": image_path, "label": label})
    return pd.DataFrame(data, columns=["image_path", "label"])

if __name__ == "__main__":
    jpg_file_paths = find_jpg_files(SEARCH_ROOT_DIR)
    df = generate_dataframe(jpg_file_paths)
    print(df.head())
    df.to_csv("annotations.csv", index=False)