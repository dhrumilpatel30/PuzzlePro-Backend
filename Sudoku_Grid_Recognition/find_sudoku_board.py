from imutils.perspective import four_point_transform
import imutils
import cv2
import numpy as np


def find_sudoku_board(image):
    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 15)
    kernel = np.ones((3, 3), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    image = cv2.dilate(image, kernel)
    threshed_image = cv2.bitwise_not(image)
    contours_list = cv2.findContours(threshed_image.copy(), cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
    contours_list = imutils.grab_contours(contours_list)
    contours_list = sorted(contours_list, key=cv2.contourArea, reverse=True)
    board_contours = None

    for c in contours_list:

        perimeter = cv2.arcLength(c, True)
        approx_border_points = cv2.approxPolyDP(c, 0.02 * perimeter, True)

        if len(approx_border_points) == 4:
            board_contours = approx_border_points
            break

    if board_contours is None:
        raise Exception("Could not found any image")

    puzzle_image = four_point_transform(image, board_contours.reshape(4, 2))
    threshed_image = four_point_transform(threshed_image, board_contours.reshape(4, 2))
    return puzzle_image, threshed_image
