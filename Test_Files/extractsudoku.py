import cv2

from Printed_Digit_Training.printed_digit_recognition import printed_digit_recognition_model, resize_to_28x28
from Sudoku_Grid_Recognition.split_digits import split_digits

current_image = cv2.imread("sudoku_test_image_1.jpg")

image_list = split_digits(current_image)
current_image1 = cv2.imread("sudoku_test_image_2.jpg")
image_list = split_digits(current_image1)
current_image2 = cv2.imread("sudoku_test_image_3.jpg")
image_list = split_digits(current_image2)
current_image4 = cv2.imread("sudoku_test_image_7.jpeg")
image_list = split_digits(current_image4)
#
# sudoku_matrix = [[0 for _ in range(9)] for _ in range(9)]
# for row_index in range(len(image_list)):
#     for col_index in range(len(image_list[row_index])):
#         processed_image = image_list[row_index][col_index]
#         processed_image = cv2.resize(processed_image, (28, 28), interpolation=cv2.INTER_AREA)
#         processed_image = resize_to_28x28(processed_image)
#         processed_image = processed_image.reshape((1, 28, 28, 1))
#         saving_image = processed_image.reshape((28, 28))
#         cv2.imwrite("./Generated/sudoku_test_image_" + str(row_index) + str(col_index) + ".jpg", saving_image)
#         sudoku_matrix[row_index][col_index] = printed_digit_recognition_model(processed_image)
#
# for digit_row in sudoku_matrix:
#     print(digit_row)
