import numpy as np
import cv2


def resize_img(image_array, new_width, new_height):
    """
    TODO Add documentation.
    """
    return cv2.resize(image_array, (new_width, new_height), interpolation=cv2.INTER_AREA)


def match_shapes(first_img, second_image):
    first_shape, second_shape = first_img.shape, second_image.shape
    max_height, max_width = max(first_shape[0], second_shape[0]), max(first_shape[1], second_shape[1])

    if first_shape[0] != max_height or first_shape[1] != max_width:
        first_img = resize_img(first_shape, max_width, max_height)

    if second_shape[0] != max_height or second_shape[1] != max_width:
        second_image = resize_img(second_shape, max_width, max_height)
    return first_img, second_image


def vertical(top_image, bottom_image):
    return np.vstack((match_shapes(top_image, bottom_image)))


def horizontal(left_image, right_image):
    return np.hstack((match_shapes(left_image, right_image)))


