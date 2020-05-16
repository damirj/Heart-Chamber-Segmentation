import os
import os.path
import SimpleITK as sitk
import cv2
import h5py
import numpy as np


def process_img_mask(image, mask, pixel_spacing):
    # Transfer image and mask from z,y,x to y,x,z
    image = image.transpose(1, 2, 0)
    mask = mask.transpose(1, 2, 0)

    # Normalizing image values to interval [0, 1]
    image = image / 255.0

    # Changing dimension of image and mask according to pixel spacing
    # dimension = (int(pixel_spacing[1] * image.shape[1] * 2), int(pixel_spacing[0] * image.shape[0] * 2))
    dimension = (256, 256)
    image = cv2.resize(image, dimension, interpolation=cv2.INTER_AREA)
    image = np.reshape(image, (256, 256, 1))
    mask = cv2.resize(mask, dimension, interpolation=cv2.INTER_AREA)
    mask = np.reshape(mask, (256, 256, 1))

    num_classes = 4
    one_hot_mask = np.zeros((mask.shape[0], mask.shape[1], num_classes))
    for i, unique_val in enumerate(np.unique(mask)):
        for row in range(256):
            for col in range(256):
                if mask[row, col] == unique_val:
                    one_hot_mask[row, col, i] = 1

    return image, one_hot_mask


def load_img_and_mask(image_name):
    mask_name = image_name[:-4] + "_gt.mhd"

    # Reads the image and mask using SimpleITK
    itk_image = sitk.ReadImage(image_name)
    itk_mask = sitk.ReadImage(mask_name)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    image = sitk.GetArrayFromImage(itk_image)
    mask = sitk.GetArrayFromImage(itk_mask)

    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itk_image.GetSpacing())))

    prepared_img, prepared_mask = process_img_mask(image, mask, spacing[1:])

    return prepared_img, prepared_mask


def load_data():
    valid_images = ["4CH_ED.mhd", "4CH_ES.mhd"]
    path = "training"
    images = []
    masks = []
    for folder in os.listdir(path):
        for file_name in os.listdir(path + "/{}".format(folder)):
            for img in valid_images:
                if img in file_name:
                    image, mask = load_img_and_mask(path + "/{}/{}".format(folder, file_name))
                    images.append(image)
                    masks.append(mask)

    images = np.asarray(images)
    masks = np.asarray(masks)
    return images, masks


def split_data(images, masks):
    train_split = int(0.7 * len(images))
    val_split = int(0.9 * len(images))

    train_images = images[:train_split]
    val_images = images[train_split:val_split]
    test_images = images[val_split:]

    train_masks = masks[:train_split]
    val_masks = masks[train_split:val_split]
    test_masks = masks[val_split:]

    return train_images, train_masks, val_images, val_masks, test_images, test_masks


def save_to_h5(filename):
    with h5py.File(filename, "w") as out:
        out.create_dataset("x_train", data=train_images, chunks=True, compression="gzip")
        out.create_dataset("y_train", data=train_masks, chunks=True, compression="gzip")
        out.create_dataset("x_val", data=val_images, chunks=True, compression="gzip")
        out.create_dataset("y_val", data=val_masks, chunks=True, compression="gzip")
        out.create_dataset("x_test", data=test_images, chunks=True, compression="gzip")
        out.create_dataset("y_test", data=test_masks, chunks=True, compression="gzip")


filename = 'heart_chambers_4ch_data.h5'
images, masks = load_data()
train_images, train_masks, val_images, val_masks, test_images, test_masks = split_data(images, masks)
save_to_h5(filename)
