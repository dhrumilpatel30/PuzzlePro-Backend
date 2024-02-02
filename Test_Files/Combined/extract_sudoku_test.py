import cv2
import keras

from Combined_Digit_Recognition.combined_digit_recognition import combined_digit_recognition, resize_to_28x28
from Sudoku_Grid_Recognition.split_digits import split_digits

current_image1 = cv2.imread("sudoku_test_image_1.jpg")
# current_image2 = cv2.imread("sudoku_test_image_3.jpg")
# image_list = split_digits(current_image2)
# current_image4 = cv2.imread("sudoku_test_image_7.jpeg")
# image_list = split_digits(current_image4)
dummy_model = keras.models.load_model('../../Models/combined_digit_model.keras', compile=False)

# Print the summary of the model
dummy_model.summary()

# Alternatively, you can print the layers
for layer in dummy_model.layers:
    print(layer.name, layer.input_shape, layer.output_shape)

# You can also check the layer names and types
for layer in dummy_model.layers:
    print(layer.name, layer.__class__.__name__)

# If you need more detailed information about a specific layer, you can print its configuration
for layer in dummy_model.layers:
    print(layer.get_config())


def recognise_mixed_sudoku(image, mixed_digit_recognition_model):
    image_list = split_digits(image)

    sudoku_matrix = [[(0, 0) for _ in range(9)] for _ in range(9)]
    for row_index in range(len(image_list)):
        for col_index in range(len(image_list[row_index])):
            processed_image = image_list[row_index][col_index]
            processed_image = cv2.resize(processed_image, (28, 28), interpolation=cv2.INTER_AREA)
            processed_image = resize_to_28x28(processed_image)
            processed_image = processed_image.reshape((1, 28, 28, 1))
            sudoku_matrix[row_index][col_index] = (
                combined_digit_recognition(processed_image, mixed_digit_recognition_model))

    return sudoku_matrix


# print(recognise_mixed_sudoku(current_image1, digit_recognition_model))
