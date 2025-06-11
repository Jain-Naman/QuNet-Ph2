import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tensorflow.keras.preprocessing import image # type: ignore

def load_images_and_masks(image_dir, mask_dir, image_size=(256, 256)):
    image_filenames = sorted([f for f in os.listdir(image_dir)])
    mask_filenames = sorted([f for f in os.listdir(mask_dir)])

    images = []
    masks = []
    
    for img_filename, mask_filename in zip(image_filenames, mask_filenames):
        img_path = os.path.join(image_dir, img_filename)
        mask_path = os.path.join(mask_dir, mask_filename)
        
        img = image.load_img(img_path, target_size=image_size)
        img = image.img_to_array(img) / 255.0  # Normalize images
        
        mask = image.load_img(mask_path, target_size=image_size, color_mode="grayscale")
        mask = image.img_to_array(mask) / 255.0  # Normalize masks
        
        images.append(img)
        masks.append(mask)

    images = np.array(images)
    masks = np.array(masks)

    return images, masks, image_filenames, mask_filenames

def create_kfold_splits(image_filenames, mask_filenames, k=5, seed=42):
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    fold_splits = []

    image_filenames = np.expand_dims(image_filenames, axis=-1)
    mask_filenames = np.expand_dims(mask_filenames, axis=-1)

    dataframe = np.concatenate((image_filenames, mask_filenames), axis=-1)

    for train_idx, val_idx in kf.split(dataframe):
        fold_splits.append((train_idx, val_idx))
    
    return fold_splits

def save_fold_splits_to_csv(fold_splits, image_filenames, mask_filenames, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for fold, (train_idx, val_idx) in enumerate(fold_splits):
        fold_train_files = [(image_filenames[i], mask_filenames[i]) for i in train_idx]
        fold_val_files = [(image_filenames[i], mask_filenames[i]) for i in val_idx]
        
        train_df = pd.DataFrame(fold_train_files, columns=["image_filename", "mask_filename"])
        val_df = pd.DataFrame(fold_val_files, columns=["image_filename", "mask_filename"])
        
        # Save train and validation splits as CSV
        train_df.to_csv(os.path.join(output_dir, f"fold_{fold+1}_train.csv"), index=False)
        val_df.to_csv(os.path.join(output_dir, f"fold_{fold+1}_val.csv"), index=False)


