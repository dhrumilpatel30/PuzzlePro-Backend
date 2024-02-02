from load_datasets import load_printed_digit_dataset
from test_model import test_printed_digit_model
from train_model import train_printed_digit_model


def make_model():
    train_images, test_images, train_labels, test_labels, number_of_classes = load_printed_digit_dataset()

    train_printed_digit_model(train_images, train_labels, number_of_classes)

    test_printed_digit_model(test_images, test_labels, number_of_classes)


make_model()
