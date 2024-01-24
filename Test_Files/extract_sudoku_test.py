import cv2

from Sudoku_Grid_Recognition.split_digits import split_digits

current_image = cv2.imread("sudoku_test_image_2.jpg")


image_list = split_digits(current_image)

for row_index in range(len(image_list)):
    for col_index in range(len(image_list[row_index])):
        cv2.imwrite("./Generated/sudoku_test_image_" + str(row_index) + str(col_index) + ".jpg",
                    image_list[row_index][col_index])
