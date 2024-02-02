import cv2
from Printed_Digit_Training.printed_digit_recognition import printed_digit_recognition_model, resize_to_28x28
from Sudoku_Grid_Recognition.split_digits import split_digits
from Combined_Digit_Recognition.combined_digit_recognition import combined_digit_recognition


def recognise_sudoku(image, digit_recognition_model):
    image_list = split_digits(image)

    sudoku_matrix = [[0 for _ in range(9)] for _ in range(9)]
    for row_index in range(len(image_list)):
        for col_index in range(len(image_list[row_index])):
            processed_image = image_list[row_index][col_index]
            processed_image = cv2.resize(processed_image, (28, 28), interpolation=cv2.INTER_AREA)
            processed_image = resize_to_28x28(processed_image)
            processed_image = processed_image.reshape((1, 28, 28, 1))
            sudoku_matrix[row_index][col_index] = (
                printed_digit_recognition_model(processed_image, digit_recognition_model))

    return sudoku_matrix


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
