import numpy as np
import cv2
import os


def load_printed_digit_dataset():
    path = 'Datasets/Digits'
    imageDimensions = (32, 32, 3)
    batchSizeVal = 50
    epochsVal = 2
    stepsPerEpochVal = 2000

    all_images = []
    class_number = []

    digits_dir_list = os.listdir(path)
    print("Total classes detected :", len(digits_dir_list))
    number_of_classes = len(digits_dir_list)

    print("Importing classes.....")

    for digit_class in range(0, number_of_classes):
        picture_list = os.listdir(
            path + "/" + str(digit_class)
        )
        for picture in picture_list:
            current_image = cv2.imread(path + "/" + str(digit_class) + "/" + picture)
            all_images.append(current_image)
            class_number.append(digit_class)
        print(digit_class, end=" ")
    print(" ")

    print("Total images in imageList : ", len(all_images))
    print("Total classes in classLabelList : ", len(class_number))

    all_images = np.array(all_images)
    class_number = np.array(class_number)

    return all_images, class_number

