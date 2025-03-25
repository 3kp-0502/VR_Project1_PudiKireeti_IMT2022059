import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

def build_unet(input_shape=(128, 128, 3)):
    inputs = keras.Input(shape=input_shape)
    conv1 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(inputs)
    conv1 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(conv3)
    up1 = layers.UpSampling2D(size=(2, 2))(conv3)
    up1 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(up1)
    merge1 = layers.concatenate([conv2, up1])
    up2 = layers.UpSampling2D(size=(2, 2))(merge1)
    up2 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(up2)
    merge2 = layers.concatenate([conv1, up2])
    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid")(merge2)
    return keras.Model(inputs, outputs)

def preprocess_image(image_path, to_rgb=False):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    image = cv2.resize(image, (128, 128)) / 255.0
    if to_rgb:
        image = np.stack([image] * 3, axis=-1)
    return image

def apply_thresholding(image):
    _, thresh = cv2.threshold((image * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh / 255

def apply_edge_detection(image):
    edges = cv2.Canny((image * 255).astype(np.uint8), 100, 200)
    return edges / 255

def compute_metrics(gt, seg):
    intersection = np.logical_and(gt, seg).sum()
    union = np.logical_or(gt, seg).sum()
    iou = intersection / (union + 1e-6)
    dice = (2 * intersection) / (np.sum(gt) + np.sum(seg) + 1e-6)
    accuracy = np.sum(gt == seg) / gt.size
    return iou, dice, accuracy

def load_data(image_folder, mask_folder):
    images, masks = [], []
    image_filenames = sorted(os.listdir(image_folder))
    for filename in tqdm(image_filenames, desc="Loading Data"):
        img_path = os.path.join(image_folder, filename)
        mask_path = os.path.join(mask_folder, filename)
        if os.path.exists(mask_path):
            img = preprocess_image(img_path, to_rgb=True)
            mask = preprocess_image(mask_path, to_rgb=False)
            if img is not None and mask is not None:
                images.append(img)
                masks.append(mask[..., np.newaxis])
    return np.array(images, dtype=np.float32), np.array(masks, dtype=np.float32)

def split_data(X, Y, test_size=0.2):
    return train_test_split(X, Y, test_size=test_size, random_state=42)

def evaluate_and_display(images, masks, model, num_samples=5):
    preds = model.predict(images) > 0.5
    iou_unet, dice_unet, acc_unet = [], [], []
    iou_thresh, dice_thresh, acc_thresh = [], [], []
    iou_edge, dice_edge, acc_edge = [], [], []
    
    for i in range(len(images)):
        gt = masks[i].squeeze()
        seg_unet = preds[i].squeeze()
        seg_thresh = apply_thresholding(images[i][..., 0])
        seg_edge = apply_edge_detection(images[i][..., 0])
        
        iou1, dice1, acc1 = compute_metrics(gt, seg_unet)
        iou2, dice2, acc2 = compute_metrics(gt, seg_thresh)
        iou3, dice3, acc3 = compute_metrics(gt, seg_edge)
        
        iou_unet.append(iou1)
        dice_unet.append(dice1)
        acc_unet.append(acc1)
        iou_thresh.append(iou2)
        dice_thresh.append(dice2)
        acc_thresh.append(acc2)
        iou_edge.append(iou3)
        dice_edge.append(dice3)
        acc_edge.append(acc3)
        
        if i < num_samples:
            fig, axs = plt.subplots(1, 5, figsize=(15, 5))
            axs[0].imshow(images[i], cmap='gray')
            axs[0].set_title("Original Image")
            axs[1].imshow(gt, cmap='gray')
            axs[1].set_title("Ground Truth")
            axs[2].imshow(seg_thresh, cmap='gray')
            axs[2].set_title("Thresholding")
            axs[3].imshow(seg_edge, cmap='gray')
            axs[3].set_title("Edge Detection")
            axs[4].imshow(seg_unet, cmap='gray')
            axs[4].set_title("U-Net Segmentation")
            for ax in axs:
                ax.axis("off")
            plt.show()
    
    print("\n=== Overall Performance ===")
    print(f"U-Net: Mean IoU: {np.mean(iou_unet):.4f}, Mean Dice: {np.mean(dice_unet):.4f}, Mean Accuracy: {np.mean(acc_unet):.4f}")
    print(f"Thresholding: Mean IoU: {np.mean(iou_thresh):.4f}, Mean Dice: {np.mean(dice_thresh):.4f}, Mean Accuracy: {np.mean(acc_thresh):.4f}")
    print(f"Edge Detection: Mean IoU: {np.mean(iou_edge):.4f}, Mean Dice: {np.mean(dice_edge):.4f}, Mean Accuracy: {np.mean(acc_edge):.4f}")

if __name__ == "__main__":
    image_folder = r"C:\Users\sanje\Downloads\MSFD\MSFD\1\face_crop"
    mask_folder = r"C:\Users\sanje\Downloads\MSFD\MSFD\1\face_crop_segmentation"
    X, Y = load_data(image_folder, mask_folder)
    X_train, X_test, Y_train, Y_test = split_data(X, Y)
    unet = build_unet()
    unet.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    unet.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=3, batch_size=8, verbose=1)
    evaluate_and_display(X_test, Y_test, unet, num_samples=5)