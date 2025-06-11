import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

from .data_loader import load_images_and_masks


def compute_iou(y_pred, y_true):
    """ Compute IoU (Intersection over Union) score for a batch of images """
    epsilon = tf.keras.backend.epsilon()
    # Flatten the predictions and true values
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    
    # Intersection
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    # Union
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection

    return (intersection + epsilon) / (union + epsilon)


def load_fold_data(fold, image_dir, mask_dir, fold_dir, image_size):
    # Load the file lists for the current fold
    train_df = pd.read_csv(f"{fold_dir}/fold_{fold}_train.csv")
    val_df = pd.read_csv(f"{fold_dir}/fold_{fold}_val.csv")
    
    train_img_files = train_df['image_filename'].values
    train_mask_files = train_df['mask_filename'].values
    val_img_files = val_df['image_filename'].values
    val_mask_files = val_df['mask_filename'].values

    # Load images and masks for the current fold
    images, masks, image_filenames, mask_filenames = load_images_and_masks(image_dir, mask_dir, image_size=image_size)
    
    train_images = [images[i] for i in range(len(image_filenames)) if image_filenames[i] in train_img_files]
    val_images = [images[i] for i in range(len(image_filenames)) if image_filenames[i] in val_img_files]
    
    train_masks = [masks[i] for i in range(len(mask_filenames)) if mask_filenames[i] in train_mask_files]
    val_masks = [masks[i] for i in range(len(mask_filenames)) if mask_filenames[i] in val_mask_files]

    return  np.array(train_images), np.array(val_images), np.array(train_masks), np.array(val_masks)


def execute_model(model, image_dir, mask_dir, fold_dir, image_size, fold, epochs, batch_size, verbose = 2):

    X_train, X_val, y_train, y_val = load_fold_data(fold, image_dir, mask_dir, fold_dir, image_size)   

    print("="*100)
    print("Successfully loaded fold {} data".format(fold))
    print("Train size: ", X_train.shape)
    print("Val size: ", X_val.shape)
    print("="*100)

    history = model.fit(X_train, y_train, validation_data = (X_val, y_val), epochs = epochs, batch_size = batch_size, verbose = verbose)
        
    y_pred = model.predict(X_val)
        
    iou = compute_iou(y_pred, y_val)

    loss = history.history["loss"]

    return iou, loss, y_pred
