import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def gaussian_distribution(x, mu, sigma, coefficient):
    """
    Returns values for a gaussian distribution based on the input array.
    Args:
        x: Array of x values.
        mu: Mean/Median/Mode/
        sigma: Standard Deviation
        coefficient: Amplitude of gaussian distribution
    Raises:
        Exception: If any of the input parameters are not of the specified type.
    Returns:
        distribution: An array of y-values that correspond to the input parameters.
    """
    distribution = coefficient*np.exp(-(x-mu)**2/(2*sigma**2))
    return distribution


def read_image_from_file(image_path, file_bit_depth=16, original_bit_depth=12):
    """
    Given a file path, returns the corresponding image array. You may also specified the original/intended bit depth
    of said image.

    Args:
        image_path: Path to the image you want to read in.
        original_bit_depth: An integer
        file_bit_depth: An integer
    Raises:
        Exception: Nothing so far.
    Returns:
        The image array in it's intended bit depth.
    """
    return np.asarray(cv2.imread(image_path, -1)) / (2**(file_bit_depth-original_bit_depth))


def get_maximum_pixel_intensity(image_array):
    """
    Takes an image array (2D grayscale), flattens it into 1D, and finds the maximum.

    Args:
        image_array: Image as a 2D numpy array.
    Returns:
        The image intensity maximum (assuming grayscale) as an int.
    """
    return max(image_array.flatten())


def get_coordinates_of_maximum(image):
    """
    Takes a 2D (grayscale) image array and finds the coordinates of the pixel with the maximum intensity value. If
    multiple pixels share the same maximum intensity value, returns centroid of those coordinates.

    Args:
        image: Image as a 2D numpy array.
    Returns:
        The image intensity maximum (assuming grayscale) as an int.
    """
    maximum_intensity_value = get_maximum_pixel_intensity(image)
    x_coordinates = np.where(image == maximum_intensity_value)[1]
    y_coordinates = np.where(image == maximum_intensity_value)[0]
    if len(x_coordinates) > 1 or len(y_coordinates) > 1:
        return [(int(np.mean(x_coordinates)), int(np.mean(y_coordinates)))][0]
    return [(i, j) for i, j in zip(x_coordinates, y_coordinates)][0]


def get_horizontal_roi(image, maximum_coords, upper_limit_noise=10):

    horizontal_center, vertical_center = maximum_coords[0], maximum_coords[1]
    horizontal_lineout = np.asarray(image[vertical_center:vertical_center+1, :])[0]
    vertical_lineout = image[:, horizontal_center:horizontal_center+1].flatten()
    print(vertical_lineout)
    index_of_1st_pixel_above_noise_limit_h = next(x[0] for x in enumerate(horizontal_lineout) if x[1] > upper_limit_noise)
    index_of_last_pixel_above_noise_limit_flo_h = len(horizontal_lineout) - next(x[0] for x in enumerate(reversed(horizontal_lineout)) if x[1] > upper_limit_noise) -1

    index_of_1st_pixel_above_noise_limit_v = next(x[0] for x in enumerate(vertical_lineout) if x[1] > upper_limit_noise)
    index_of_last_pixel_above_noise_limit_flo_v = len(vertical_lineout) - next(x[0] for x in enumerate(reversed(vertical_lineout)) if x[1] > upper_limit_noise) -1

    fig_a, ax = plt.subplots()
    ax.title.set_text('Original Image')
    ax.imshow(image)
    fig_a.savefig("Original_Image.png")

    fig_b, bx = plt.subplots()
    bx.title.set_text('Horizontal Region of Interest (Above Noise)')
    bx.imshow(image[:, index_of_1st_pixel_above_noise_limit_h:index_of_last_pixel_above_noise_limit_flo_h])
    fig_b.savefig("Region_of_Interest_Horizontal.png")


    fig_c, cx = plt.subplots()
    cx.title.set_text('Horizontal & Vertical Region of Interest (Above Noise)')
    cx.imshow(image[index_of_1st_pixel_above_noise_limit_v:index_of_last_pixel_above_noise_limit_flo_v, index_of_1st_pixel_above_noise_limit_h:index_of_last_pixel_above_noise_limit_flo_h])
    fig_c.savefig("Region_of_Interest_Horizontal_Vertical.png")


def plot_horizonal_lineout_intensity(image, coordinates, upper_limit_noise=10):
    """

    Args:
        image: The image as a 2D Array
        coordinates: The coordinates you want a horizontal line-out of.
    """
    horizontal_center, vertical_center = coordinates[0], coordinates[1]
    horizontal_lineout = np.asarray(image[vertical_center:vertical_center+1, :])[0]
    index_of_1st_pixel_above_noise_limit = next(x[0] for x in enumerate(horizontal_lineout) if x[1] > upper_limit_noise)

    indices = np.arange(0, len(horizontal_lineout))
    fig = plt.figure()
    plt.plot(indices, horizontal_lineout)
    plt.title("Full Line-out")
    plt.show()
    fig.savefig("Full_Line_out.png")
    plt.close('all')

    left_trimmed_lineout = horizontal_lineout[index_of_1st_pixel_above_noise_limit:]
    left_trimmed_indices = np.arange(0, len(left_trimmed_lineout))
    fig2 = plt.figure()
    plt.plot(left_trimmed_indices, left_trimmed_lineout)
    plt.title("Left Trimmed Line-out")
    plt.show()
    fig2.savefig("Left_Trimmed_Line_out.png")
    plt.close('all')


    index_of_last_pixel_above_noise_limit = len(left_trimmed_lineout) - next(x[0] for x in enumerate(reversed(left_trimmed_lineout)) if x[1] > upper_limit_noise) -1

    right_trimmed_lineout = left_trimmed_lineout[:index_of_last_pixel_above_noise_limit+1]
    right_trimmed_indices = np.arange(0, len(right_trimmed_lineout))
    fig3 = plt.figure()
    plt.plot(right_trimmed_indices, right_trimmed_lineout)
    plt.title("Left-Right Trimmed Line-out")
    plt.show()
    fig3.savefig("Left_Right_Trimmed_Line_out.png")
    plt.close('all')

    height, width = image.shape[0], image.shape[1]




#img = read_image_from_file("./coregistration/cam_b_frame_186.png")

#get_horizontal_roi(img, get_coordinates_of_maximum(img))
