from utils.data_loader import save_fold_splits_to_csv, load_images_and_masks, create_kfold_splits
import os


IMAGE_SIZE = (192, 256)
k = 5

image_dir = 'ph2_dataset/images/'
mask_dir = 'ph2_dataset/masks/'
fold_dir = 'ph2_dataset/folds/'

if __name__ == "__main__":
    images, masks, image_filenames, mask_filenames = load_images_and_masks(image_dir, mask_dir, image_size = IMAGE_SIZE)
    fold_splits = create_kfold_splits(image_filenames, mask_filenames, k=k, seed=42)
    save_fold_splits_to_csv(fold_splits, image_filenames, mask_filenames, fold_dir)
    print("Save successful")