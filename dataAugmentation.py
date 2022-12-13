# Usage :-
# To Augment the images for specified folder names in list folders
# python .\dataAugmentation.py

import cv2
import os
import random
import numpy as np


def load_images_from_folder(folder):
    images = []
    image_counter = 0
    for filename in os.listdir(folder):
        if "flip" not in filename:
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                images.append(img)
    return images


def horizontal_flip_image(images, folder):
    print(len(images))
    image_counter = 0
    for img in images:
        horizontal_image = cv2.flip(img, 1)
        cv2.imwrite(
            folder+"/horizontal_image_flip_{}.png".format(image_counter), horizontal_image)
        image_counter += 1


def vertical_flip_image(images, folder):
    print(len(images))
    image_counter = 0
    for img in images:
        vertical_image = cv2.flip(img, 0)
        cv2.imwrite(
            folder+"/verticaltal_image_flip_{}.png".format(image_counter), vertical_image)
        image_counter += 1


def rotate_image(images, folder, angle):
    print(len(images))
    image_counter = 0
    for img in images:
        image_center = tuple(np.array(img.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(
            img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
        cv2.imwrite(
            folder+"/rotated_image_{}.png".format(image_counter), result)
        image_counter += 1


def guassian_blur_image(images, folder):
    print(len(images))
    image_counter = 0
    for img in images:
        i = random.randrange(7, 11, 2)
        guassian_blur_image = cv2.GaussianBlur(img, (i, i), 1)
        cv2.imwrite(
            folder+"/guassain_blur_{}.png".format(image_counter), guassian_blur_image)
        image_counter += 1


folders = ["Sagar-Face", "Aditya-Face"]

angle = random.randint(-20, 20)

for folder in folders:
    horizontal_flip_image(load_images_from_folder(
        "C:/Users/Akshay Shrivastava/OneDrive/Desktop/BTP/datasets/train/" + folder), "C:/Users/Akshay Shrivastava/OneDrive/Desktop/BTP/datasets/train/" + folder)
    vertical_flip_image(load_images_from_folder(
        "C:/Users/Akshay Shrivastava/OneDrive/Desktop/BTP/datasets/train/" + folder), "C:/Users/Akshay Shrivastava/OneDrive/Desktop/BTP/datasets/train/" + folder)
    rotate_image(load_images_from_folder("C:/Users/Akshay Shrivastava/OneDrive/Desktop/BTP/datasets/train/" +
                 folder), "C:/Users/Akshay Shrivastava/OneDrive/Desktop/BTP/datasets/train/" + folder, angle)
    guassian_blur_image(load_images_from_folder(
        "C:/Users/Akshay Shrivastava/OneDrive/Desktop/BTP/datasets/train/" + folder), "C:/Users/Akshay Shrivastava/OneDrive/Desktop/BTP/datasets/train/" + folder)
