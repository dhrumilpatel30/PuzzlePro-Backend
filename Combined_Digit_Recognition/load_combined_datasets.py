import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
import cv2
import numpy as np
from keras.src.utils import to_categorical


def normalize_image(current_image):
    current_image = cv2.equalizeHist(current_image)
    current_image = current_image / 255
    return current_image


def load_combined_datasets():
    path = 'C:/Users/user1/digitmodel/PuzzlePro-Backend/Datasets/Digits'
    train_size, test_size = 0.80, 0.20

    all_images = []
    class_number = []

    digits_dir_list = os.listdir(path)
    print("Total classes detected:", len(digits_dir_list))
    number_of_classes = len(digits_dir_list)

    print("Importing classes for training...")

    for digit_class in range(0, number_of_classes):
        picture_list = os.listdir(path + "/" + str(digit_class))
        for picture in picture_list:
            current_image = cv2.imread(path + "/" + str(digit_class) + "/" + picture, cv2.IMREAD_GRAYSCALE)
            all_images.append(current_image)
            class_number.append(digit_class)
        print(f"Class {digit_class}: {len(picture_list)} images imported.")

    print(" ")
    
    all_images = np.array(all_images)
    class_number = np.array(class_number)
    # Splitting printed digits dataset
    (printed_train_images, printed_test_images,
     printed_train_labels, printed_test_labels) = (
        train_test_split(all_images, class_number,
                         train_size=train_size, test_size=test_size, random_state=30))


    printed_train_images = np.array(list(map(normalize_image, printed_train_images)))
    printed_test_images = np.array(list(map(normalize_image, printed_test_images)))

    # Load MNIST dataset
    (mnist_train_images, mnist_train_labels), (mnist_test_images, mnist_test_labels) = mnist.load_data()
    mnist_mask_train = mnist_train_labels != 0
    mnist_mask_test = mnist_test_labels != 0
    mnist_train_images = mnist_train_images[mnist_mask_train]
    mnist_train_labels = mnist_train_labels[mnist_mask_train]
    mnist_test_images = mnist_test_images[mnist_mask_test]
    mnist_test_labels = mnist_test_labels[mnist_mask_test]
    
    

    # Combining printed digits dataset with MNIST training and test sets
    combined_images_train = np.concatenate((printed_train_images, mnist_train_images))
    combined_labels_train = np.concatenate((printed_train_labels, mnist_train_labels))
    combined_images_test = np.concatenate((printed_test_images, mnist_test_images))
    combined_labels_test = np.concatenate((printed_test_labels, mnist_test_labels))

    combined_labels_encoded_train = to_categorical(combined_labels_train, num_classes=19)
    combined_labels_encoded_test = to_categorical(combined_labels_test, num_classes=19)


    combined_images_train = (
        combined_images_train.reshape(combined_images_train.shape[0], combined_images_train.shape[1],
                                      combined_images_train.shape[2], 1))

    print("Dataset Splitting Complete.")
    print(f"Number of training images: Printed Digits - {len(printed_train_images)}, MNIST - {len(mnist_train_images)}")
    print(f"Number of testing images: Printed Digits - {len(printed_test_images)}, MNIST - {len(mnist_test_images)}")

    print("Data Combination Complete.")

    return combined_images_train, combined_images_test, combined_labels_encoded_train, combined_labels_encoded_test, 19
