import time
import cv2
import numpy as np


def resize_to_28x28(image):
    image_height, image_width = image.shape
    dim_size_max = max(image.shape)

    if dim_size_max == image_width:
        im_h = (26 * image_height) // image_width
        if im_h <= 0 or image_width <= 0:
            print("Invalid Image Dimension: ", im_h, image_width, image_height)
        tmp_img = cv2.resize(image, (26, im_h), 0, 0, cv2.INTER_NEAREST)
    else:
        im_w = (26 * image_width) // image_height
        if im_w <= 0 or image_height <= 0:
            print("Invalid Image Dimension: ", im_w, image_width, image_height)
        tmp_img = cv2.resize(image, (im_w, 26), 0, 0, cv2.INTER_NEAREST)

    out_img = np.zeros((28, 28), dtype=np.ubyte)

    nb_h, nb_w = out_img.shape
    na_h, na_w = tmp_img.shape
    y_min = nb_w // 2 - (na_w // 2)
    y_max = y_min + na_w
    x_min = nb_h // 2 - (na_h // 2)
    x_max = x_min + na_h

    out_img[x_min:x_max, y_min:y_max] = tmp_img

    return out_img


def combined_digit_recognition(digit_image, combined):
    start_time = time.time()
    predictions = combined.predict(digit_image)
    predicted_digit = np.argmax(predictions)
    print("---Took time %s seconds ---" % (time.time() - start_time))
    print(predicted_digit)
    print(predictions)
    return predicted_digit
