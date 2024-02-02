import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split


def load_printed_digit_dataset():
    path = 'C:/Users/user1/digitmodel/PuzzlePro-Backend/Datasets/Digits'
    train_size, test_size = 0.80, 0.20

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

    train_images, test_images, train_labels, test_labels = train_test_split(all_images, class_number,
                                                                            train_size=train_size, test_size=test_size)

    del all_images, class_number

    return train_images, test_images, train_labels, test_labels, number_of_classes
