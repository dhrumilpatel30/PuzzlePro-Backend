from test_combined_model import test_combined_digit_model
from train_combined_model import train_combined_digit_model
from load_combined_datasets import load_combined_datasets


def make_combined_model():
    train_images, test_images, train_labels, test_labels, number_of_classes = load_combined_datasets()

    # train_combined_digit_model(train_images, test_images, train_labels, test_labels, number_of_classes)

    test_combined_digit_model(test_images, test_labels)


make_combined_model()
