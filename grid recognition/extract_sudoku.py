import cv2
from grid_recogniser import grid_recognition


def exact_sudoku():
    image = cv2.imread('../sample images/sudoku1.jpg')
    grid = grid_recognition(image)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


exact_sudoku()
