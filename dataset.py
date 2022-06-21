import os

import cv2
import numpy as np
import pandas as pd

import itk
from patchify import patchify
import albumentations as alb


def normalize_color(image, reference):
    ImageType = itk.Image[itk.RGBPixel[itk.UC], 2]
    image_itk = itk.image_from_array(image, ttype=ImageType)
    reference_itk = itk.image_from_array(reference, ttype=ImageType)

    normalized_img = itk.structure_preserving_color_normalization_filter(
        image_itk,
        reference_itk,
        color_index_suppressed_by_hematoxylin=0,
        color_index_suppressed_by_eosin=1)

    return np.asarray(normalized_img)


def augment_image(image, mask, number_of_augmentations,
                  transform = alb.Compose([
                    alb.HorizontalFlip(p=0.5),
                    alb.VerticalFlip(p=0.5),
                    alb.RandomRotate90(p=0.5),
                  ]),
                  max_rotation=20):

    images = []
    masks = []

    for i in range(number_of_augmentations):
        transformed = transform(image=image, mask=mask)
        images.append(transformed['image'])
        masks.append(transformed['mask'])

    return images, masks


def patchify_image(image, mask, shape):
    images_shape = (*shape, 3)
    masks_shape = (*shape, 1)
    image_patches = patchify(image, images_shape, step=128)
    mask_patches = patchify(mask, masks_shape, step=128)

    num_patches = np.prod(np.array(image_patches.shape[:-3]))
    image_patches = np.reshape(image_patches, (num_patches, *images_shape))
    mask_patches = np.reshape(mask_patches, (num_patches, *masks_shape))
    return image_patches, mask_patches


def read_augment_dataset(data_folder: str, output_folder: str, number_of_augmentations: int):
    df = pd.read_csv(os.path.join(data_folder, "grade.csv"))
    metadata = []
    reference_image = None

    for index, row in df.iterrows():
        print(f"Processing image {index} of {df.shape[0]}", end='\r')
        img_file_name = f"{row['name']}.bmp"
        mask_file_name = f"{row['name']}_anno.bmp"

        # read image and apply soft blurring
        image = cv2.imread(os.path.join(data_folder, img_file_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.medianBlur(image, 3)

        if index == 0:
            reference_image = image
        else:
            image = normalize_color(image, reference_image)

        # read mask
        mask = cv2.imread(os.path.join(data_folder, mask_file_name))

        # after this call, masks are reduced to (256, 256, 1)
        images_patches, mask_patches = patchify_image(image, mask, (256, 256))

        for i in range(images_patches.shape[0]):
            augmented_images, augmented_masks = augment_image(images_patches[i], mask_patches[i],
                                                              number_of_augmentations=number_of_augmentations)

            for j in range(number_of_augmentations):
                image_name = f"{row['name']}-p_{i}-a_{j}.bmp"
                mask_name = f"{row['name']}-p_{i}-a_{j}_anno.bmp"
                cv2.imwrite(os.path.join(output_folder, image_name), augmented_images[j])
                cv2.imwrite(os.path.join(output_folder, mask_name), augmented_masks[j])

                metadata.append({
                    'name': row['name'],
                    'image': image_name,
                    'mask': mask_name,
                    'patient': row['patient ID'],
                    'grade1': row[' grade (GlaS)'],
                    'grade2': row[' grade (Sirinukunwattana et al. 2015)']
                })

    df_processed = pd.DataFrame(metadata)
    df_processed.to_csv(os.path.join(output_folder, 'grade.csv'))


if __name__ == "__main__":
    folder = "dataset"
    output = "augmented_dataset"
    read_augment_dataset(data_folder=folder, output_folder=output, number_of_augmentations=3)
