from utils.kfold import execute_model, compute_iou
from models.qunet_12_2 import QuNet

import numpy as np
import warnings

warnings.filterwarnings("ignore", message="You are casting an input of type complex128 *")

IMAGE_SIZE = (192, 256)
BATCH_SIZE = 18
EPOCHS = 20
k = 5
c_last_input = 8
q_filters = 1
splits = int(c_last_input/q_filters)

image_dir = 'ph2_dataset/images/'
mask_dir = 'ph2_dataset/masks/'
fold_dir = 'ph2_dataset/folds/'

def cross_validate(num_filters):
    iou_scores = []

    for fold in range(3, k+1):
        model = QuNet((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), compute_iou, num_filters, splits = splits)

        if fold == 1:
            print(model.summary()) 

        iou, loss, predictions = execute_model(model, image_dir, mask_dir, fold_dir, IMAGE_SIZE, fold, EPOCHS, BATCH_SIZE, verbose=2)
        
        print("Fold {} - IoU {} - Loss {}".format(fold, iou, loss))
        iou_scores.append(iou)

        np.save("results/qunet/qunet_12_2/predictions_fold{}.npy".format(fold), predictions)

    avg_iou = np.mean(iou_scores)
    return avg_iou


def main():

    num_filters = [8, 16, 32, 16, c_last_input]

    avg_iou_qunet = cross_validate(num_filters)

    print("="*100)
    print("="*100)
    print(f"Average IoU: {avg_iou_qunet:.4f}")
    print("="*100)
    print("="*100)

if __name__ == "__main__":
    main()