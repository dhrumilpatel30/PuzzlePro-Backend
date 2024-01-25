from imutils.perspective import four_point_transform
import imutils
import cv2


def find_sudoku_board(image, debug=False):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (7, 7), 3)

    threshed_image = cv2.adaptiveThreshold(blurred_image, 255,
                                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    threshed_image = cv2.bitwise_not(threshed_image)

    if debug:
        cv2.imshow("Puzzle Thresh", threshed_image)
        cv2.waitKey(0)

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

    if debug:
        output = image.copy()
        cv2.drawContours(output, [board_contours], -1, (0, 255, 0), 2)
        cv2.imshow("Puzzle Outline", output)
        cv2.waitKey(0)

    puzzle_image = four_point_transform(image, board_contours.reshape(4, 2))
    threshed_image = four_point_transform(threshed_image, board_contours.reshape(4, 2))
    if debug:
        cv2.imshow("Puzzle Transform", puzzle_image)
        cv2.waitKey(0)

    cv2.imwrite("./Generated/sudoku_test_image_board.jpg", threshed_image)

    return puzzle_image, threshed_image
