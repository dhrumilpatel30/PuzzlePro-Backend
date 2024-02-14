from Sudoku_Grid_Recognition.find_sudoku_board import find_sudoku_board


def split_digits(image):
    puzzle_image, warped_image = find_sudoku_board(image)

    number_of_rows = 9
    number_of_cols = 9
    padding = (3, 3)

    split_image_matrix = [[0 for _ in range(number_of_cols)] for _ in range(number_of_rows)]
    image_width = warped_image.shape[1] // number_of_rows
    image_height = warped_image.shape[0] // number_of_cols

    for row_index in range(number_of_rows):
        lower_y_coordinate = row_index * image_height + padding[1]
        higher_y_coordinate = (row_index + 1) * image_height - padding[1]

        for col_index in range(number_of_cols):
            lower_x_coordinate = col_index * image_width + padding[0]
            higher_x_coordinate = (col_index + 1) * image_width - padding[0]

            split = warped_image[lower_y_coordinate:higher_y_coordinate, lower_x_coordinate:higher_x_coordinate]

            split_image_matrix[row_index][col_index] = split

    return split_image_matrix
