



import os
# from cda_comp.src.baseline_simple_cnn import train_and_evaluate
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset, WeightedRandomSampler
from torchvision import transforms, models
import timm  # For ViT
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
# Data transforms
from torchvision.transforms import GaussianBlur, ColorJitter, RandomErasing
from sklearn.metrics import confusion_matrix
import random
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.transforms import RandAugment, RandomAffine
from tqdm import tqdm
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
csv_file = '/data/hma18/CDA_hackathon/FBSI/annotations.csv'
root_dir = '/data/hma18/CDA_hackathon/FBSI/dataset'
outputs_dir = "/data/hma18/CDA_hackathon/cda_comp/outputs"
# train_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.RandomHorizontalFlip(),
#     # transforms.ColorJitter(brightness=0.5, contrast=0.5),  # Simulate low/high light
#     # GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),  # Random blur
#     transforms.ToTensor(),
#     # transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),  # Partial occlusion
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     # lambda x: x + torch.randn_like(x) * 0.1  # Mild Gaussian noise
# ])
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),                # larger resize
    transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15),
    transforms.RandomAffine(degrees=0, shear=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.4, scale=(0.02, 0.2)),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Custom Dataset
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import random

# def create_gabor_kernel(ksize, sigma, theta, lambd, gamma, psi):
#     """Create a Gabor kernel with specified parameters"""
#     return cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
# def get_adaptive_gabor_filtered_image(image, output_image_path=None):
#     """
#     Apply adaptive Gabor filtering to an image and save the result.
    
#     Parameters:
#     - input_image_path: Path to the input image
#     - output_image_path: Path to save the filtered image (optional, auto-generated if None)
    
#     Returns:
#     - output_path: Path where the filtered image was saved
#     - success: Boolean indicating if the operation was successful
#     """
#     try:
       
#         # Load image
#         # image = cv2.imread(image)
#         if image is None:
#             print(f"Error: Could not load image from '{image}'")
#             return None
        
#         # Convert to grayscale if needed
#         if len(image.shape) == 3:
#             # print("Converting to grayscale")
#             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         else:
#             # print("Image is already grayscale")
#             gray = image
        
#         # Optimized parameters for adaptive Gabor filtering
#         orientations = np.arange(0, 180, 22.5)  # 8 orientations
#         wavelengths = [8, 16, 24]  # Multiple scales
#         sigma = 3  # Standard deviation
#         gamma = 0.7  # Aspect ratio
#         psi = 0  # Phase offset
#         ksize = 21  # Kernel size
        
#         # Convert orientations to radians
#         orientations_rad = np.deg2rad(orientations)
        
#         responses = []
        
#         # Apply Gabor filters
#         for wavelength in wavelengths:
#             for theta in orientations_rad:
#                 kernel = create_gabor_kernel(ksize, sigma, theta, wavelength, gamma, psi)
#                 response = cv2.filter2D(gray, cv2.CV_32F, kernel)
#                 responses.append(response)
        
#         responses = np.array(responses)
        
#         # Adaptive method: Combined magnitude response with adaptive thresholding
#         combined = np.sqrt(np.sum(responses**2, axis=0))
        
#         # Normalize to 0-255 range
#         combined_norm = ((combined - combined.min()) / (combined.max() - combined.min()) * 255).astype(np.uint8)
        
#         # Apply adaptive thresholding for final result
#         adaptive_result = cv2.adaptiveThreshold(
#             combined_norm, 255, 
#             cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#             cv2.THRESH_BINARY, 11, 2
#         )
        
#         # Post-processing: Apply morphological operations to clean up
#         kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#         final_result = cv2.morphologyEx(adaptive_result, cv2.MORPH_CLOSE, kernel_morph)
        
#         # cv2.imwrite("adaptive_gabor_filtered_image_cleaned_labeled.png", final_result)
        
#         return final_result
            
#     except Exception as e:
#         print(f"Error processing image : {e}")
#         return None


# def create_gabor_kernel(ksize, sigma, theta, lambd, gamma, psi):
#     return cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)


# def get_fast_gabor_filtered_image(image):
#     """Fast version: ~5-8× faster than original"""
#     try:
#         if image is None:
#             return None

#         if len(image.shape) == 3:
#             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         else:
#             gray = image.astype(np.float32)

#         # Fast parameters
#         # orientations = np.deg2rad([0, 45, 90, 135])  # 4 directions
#         # wavelengths = [10, 20]                       # 2 scales
#         # sigma = 3.0
#         # gamma = 0.8
#         # psi = 0
#         # ksize = 13

#         orientations = np.deg2rad([0, 45, 90, 135])     # 4 main directions
#         wavelengths  = [8, 14]                          # lower wavelengths = finer/stronger edges
#         sigma        = 2.8                              # tighter envelope → sharper response
#         gamma        = 0.5                              # elongated kernels → stronger directionality
#         psi          = 0
#         ksize        = 15                               # good balance speed vs. strength

#         responses = []
#         for wl in wavelengths:
#             for theta in orientations:
#                 kernel = create_gabor_kernel(ksize, sigma, theta, wl, gamma, psi)
#                 resp = cv2.filter2D(gray.astype(np.float32), cv2.CV_32F, kernel)
#                 responses.append(np.abs(resp))

#         combined = np.sqrt(np.sum(np.square(responses), axis=0))

#         if combined.max() > combined.min():
#             combined_norm = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#         else:
#             combined_norm = np.zeros_like(combined, dtype=np.uint8)

#         return combined_norm

#     except Exception as e:
#         print(f"Gabor error: {e}")
#         return None
    
def get_extreme_fast_gabor(image, downscale_factor=0.4):
    """
    Extreme speed version – good when processing thousands of images per epoch
    - Downscale first → huge speedup
    - Only 4 filters
    - Almost no post-processing
    """
    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.astype(np.float32)

    # Downscale aggressively
    h, w = gray.shape
    small = cv2.resize(gray, None, fx=downscale_factor, fy=downscale_factor, interpolation=cv2.INTER_AREA)

    kernels = [
        cv2.getGaborKernel((9,9), 2.2, 0.0,     9, 0.65, 0, cv2.CV_32F),
        cv2.getGaborKernel((9,9), 2.2, np.pi/4, 9, 0.65, 0, cv2.CV_32F),
        cv2.getGaborKernel((9,9), 2.2, np.pi/2, 9, 0.65, 0, cv2.CV_32F),
        cv2.getGaborKernel((9,9), 2.2, 3*np.pi/4,9, 0.65, 0, cv2.CV_32F),
    ]

    responses = [np.abs(cv2.filter2D(small.astype(np.float32), cv2.CV_32F, k)) for k in kernels]
    combined = np.sqrt(np.mean(np.square(responses), axis=0))

    # Quick normalize
    if combined.max() > combined.min():
        norm = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        norm = np.zeros_like(combined, dtype=np.uint8)

    # Upscale back to original size
    result = cv2.resize(norm, (w, h), interpolation=cv2.INTER_LINEAR)

    return result
class FeedBunkDataset(Dataset):
    def __init__(self, annotations_df, root_dir, transform=None, gabor_prob=0.5):
        """
        Args:
            gabor_prob (float): Probability of applying Gabor filtering (0.0–1.0)
                                Set to 0.0 to disable, >0.0 for augmentation
        """
        self.annotations = annotations_df
        self.root_dir = root_dir
        self.transform = transform
        self.gabor_prob = gabor_prob
        
        self.unique_scores = sorted(self.annotations['score'].unique())
        self.score_to_label = {score: idx for idx, score in enumerate(self.unique_scores)}
        self.label_to_score = {idx: score for score, idx in self.score_to_label.items()}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = self.annotations.iloc[idx]['name_id'] + '.jpg'  # ← note: you changed to .jpg
        img_path = None
        for subdir in os.listdir(self.root_dir):
            candidate_path = os.path.join(self.root_dir, subdir, img_name)
            if os.path.exists(candidate_path):
                img_path = candidate_path
                break
        if img_path is None:
            raise FileNotFoundError(f"Image {img_name} not found in {self.root_dir}")

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Failed to load {img_path}")
        
        # Convert BGR → RGB (cv2 loads BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Decide whether to apply Gabor (only during training)
        apply_gabor = random.random() < self.gabor_prob
        # apply_gabor = True

        if apply_gabor:
            # Apply your adaptive Gabor function
            # gabor_result = get_adaptive_gabor_filtered_image(image)
            # gabor_result = get_fast_gabor_filtered_image(image)
            gabor_result = get_extreme_fast_gabor(image)
            if gabor_result is not None:
                # Gabor returns binary-like image → convert to 3-channel for consistency
                gabor_3ch = cv2.cvtColor(gabor_result, cv2.COLOR_GRAY2RGB)
                image = gabor_3ch
            # else: keep original if failed

        # Convert back to PIL for torchvision transforms
        image = Image.fromarray(image)

        label = self.score_to_label[self.annotations.iloc[idx]['score']]

        if self.transform:
            image = self.transform(image)

        return image, label

    def save_previews(self, count=10, save_dir=None, apply_gabor_preview=True):
        """Save preview images — optionally with Gabor applied"""
        if save_dir is None:
            save_dir = os.path.join("outputs", "previews")
        os.makedirs(save_dir, exist_ok=True)

        for i in tqdm(range(min(count, len(self))), desc="Saving Previews"):
            # Temporarily force Gabor for preview if requested
            orig_prob = self.gabor_prob
            if apply_gabor_preview:
                self.gabor_prob = 1.0

            image, label = self[i]
            self.gabor_prob = orig_prob  # restore

            if isinstance(image, torch.Tensor):
                image = transforms.ToPILImage()(image)
            
            score = self.label_to_score[label]
            img_name = self.annotations.iloc[i]['name_id']
            suffix = "_with_gabor" if apply_gabor_preview else ""
            image.save(os.path.join(save_dir, f"{img_name}_score_{score}{suffix}.jpg"))


# Load annotations
df = pd.read_csv(csv_file)

# ──────────────────────────────────────────────
# Step 1: Merge label 0.0 and 0.5 into 0.5
# ──────────────────────────────────────────────
df['score'] = df['score'].replace({0.0: 0.5})

# print("After merging 0.0 → 0.5:")
# print(df['score'].value_counts().sort_index())

# ──────────────────────────────────────────────
# Step 2: Reduce label 4 to exactly 300 samples
#         Prefer removing from farms 0, 1, 5
# ──────────────────────────────────────────────
label4 = df[df['score'] == 4.0].copy()
current_count = len(label4)

# print(f"\nCurrent total label 4 samples: {current_count}")

if current_count > 300:
    to_remove = current_count - 300
    
    # Priority: remove from farms 0,1,5
    priority_farms = [0, 1, 5]
    priority_label4 = label4[label4['farm'].isin(priority_farms)]
    
    if len(priority_label4) >= to_remove:
        # Enough in priority farms → remove only from them
        remove_idx = priority_label4.sample(n=to_remove, random_state=42).index
    else:
        # Remove all from priority farms + random from others
        remove_from_priority = priority_label4.index
        remaining_to_remove = to_remove - len(remove_from_priority)
        
        other_label4 = label4[~label4.index.isin(remove_from_priority)]
        remove_from_other = other_label4.sample(n=remaining_to_remove, random_state=42).index
        
        remove_idx = remove_from_priority.union(remove_from_other)
    
    # Drop the selected rows
    df = df.drop(remove_idx)
    
    # print(f"Removed {to_remove} samples of label 4 (prioritized farms 0,1,5)")
else:
    print("No removal needed (label 4 already ≤ 300)")

# print("Final label counts:")
# print(df['score'].value_counts().sort_index())

# ──────────────────────────────────────────────
# Step 3: Farm-based split (after merging & reduction)
# ──────────────────────────────────────────────
train_farms = [5, 7, 1, 0, 4, 8]
val_farms   = [3, 6]
test_farms  = [2]

train_df = df[df['farm'].isin(train_farms)].copy()
val_df   = df[df['farm'].isin(val_farms)].copy()
test_df  = df[df['farm'].isin(test_farms)].copy()

# print("\nTrain set label distribution:")
# print(train_df['score'].value_counts().sort_index())
# print(f"→ Label 4 in train: {(train_df['score'] == 4.0).sum()}")

# print("\nVal set label distribution:")
# print(val_df['score'].value_counts().sort_index())

# print("\nTest set label distribution:")
# print(test_df['score'].value_counts().sort_index())

# Create datasets
train_dataset = FeedBunkDataset(train_df, root_dir, transform=train_transform, gabor_prob=1.0)
val_dataset = FeedBunkDataset(val_df, root_dir, transform=test_transform, gabor_prob=1.0)
test_dataset = FeedBunkDataset(test_df, root_dir, transform=test_transform, gabor_prob=1.0)



if __name__ == "__main__":
    # Save preview images for sanity check
    train_dataset.save_previews(count=800, save_dir="./train_previews_strong_gabor", apply_gabor_preview=True)