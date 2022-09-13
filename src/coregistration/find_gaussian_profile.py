import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from PIL import Image

from exceptions import  coregistration_exceptions as cre

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
    # distribution = coefficient*np.exp(-(x-mu)**2/(2*sigma**2))
    return coefficient * np.exp(-(x - mu) ** 2 / (2.0 * sigma ** 2))
    # return distribution


# Create a function which returns a Gaussian (normal) distribution.
def gauss(x, *p):
    a, b, c, d = p
    y = a * np.exp(-np.power((x - b), 2.) / (2. * c ** 2.))  # + d
    return y


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
    return cv2.imread(image_path, -1) / (2 ** (file_bit_depth - original_bit_depth))
    # return np.asarray(cv2.imread(image_path, -1)) / (2**(file_bit_depth-original_bit_depth))


def get_maximum_pixel_intensity(image_array):
    """
    Takes an image array (2D grayscale), flattens it into 1D, and finds the maximum.

    Args:
        image_array: Image as a 2D numpy array.
    Returns:
        The image intensity maximum (assuming grayscale) as an int.
    """
    return max(image_array.flatten().tolist())


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
        x, y = [(int(np.mean(x_coordinates)), int(np.mean(y_coordinates)))][0]
    else:
        x, y = [(i, j) for i, j in zip(x_coordinates, y_coordinates)][0]
    return x, y


def crop_surrounding_noise(image, maximum_coords, upper_limit_noise=10):
    """
    Given an image and the coordinates of the maximum intensity, crops out noise surrounding signal.

    Args:
        image:
        maximum_coords:
        upper_limit_noise: Pixels with intensity values less than or equal to this noise limit will be cropped out from
        the image.
    Returns:
        np.ndarray: Newly cropped image
    """

    # Steps of cropping surrounding noise
    horizontal_center, vertical_center = maximum_coords[0], maximum_coords[1]
    horizontal_lineout = np.asarray(image[vertical_center:vertical_center + 1, :])[0]
    vertical_lineout = image[:, horizontal_center:horizontal_center + 1].flatten()
    index_of_1st_pixel_above_noise_limit_h = next(
        x[0] for x in enumerate(horizontal_lineout) if x[1] > upper_limit_noise)
    index_of_last_pixel_above_noise_limit_flo_h = len(horizontal_lineout) - next(
        x[0] for x in enumerate(reversed(horizontal_lineout)) if x[1] > upper_limit_noise) - 1

    index_of_1st_pixel_above_noise_limit_v = next(x[0] for x in enumerate(vertical_lineout) if x[1] > upper_limit_noise)
    index_of_last_pixel_above_noise_limit_flo_v = len(vertical_lineout) - next(
        x[0] for x in enumerate(reversed(vertical_lineout)) if x[1] > upper_limit_noise) - 1

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
    cx.imshow(image[index_of_1st_pixel_above_noise_limit_v:index_of_last_pixel_above_noise_limit_flo_v,
              index_of_1st_pixel_above_noise_limit_h:index_of_last_pixel_above_noise_limit_flo_h])
    fig_c.savefig("Region_of_Interest_Horizontal_Vertical.png")
    plt.close('all')
    return image[index_of_1st_pixel_above_noise_limit_v:index_of_last_pixel_above_noise_limit_flo_v,
           index_of_1st_pixel_above_noise_limit_h:index_of_last_pixel_above_noise_limit_flo_h]


def plot_horizonal_lineout_intensity(image, coordinates):
    """
    Finds the coordinates (x_max, y_max) of the pixel with the maximum intensity, and plots a line-out of intensity
    values for pixels where y=y_max.

    Args:
        image: The image as a 2D Array
        coordinates: Coordinates you'd like a line-out of.
    Returns:
        (np.ndarray, np.ndarray): A tuple equal to (indices, intensity values). Both components as a np.ndarray
    """
    # print("Inside Function plot_horizonal_lineout_intensity(image, coordinates)")

    fig = plt.figure()
    y_max, x_max = coordinates

    lineout = np.asarray(image[x_max:x_max + 1, :])[0]
    indices = np.arange(0, len(lineout))
    # print("Lineout Shape: ", lineout.shape)
    plt.plot(indices, lineout)
    plt.title("Left-Right Trimmed Line-out")
    # plt.show()
    plt.show()
    plt.close('all')
    fig.savefig("Lineout.png")
    return indices, lineout


def fit_function(x_coords, y_coords):
    fig = plt.figure()
    print("Scipy Optimize Curve Fit Parameters")
    popt, pcov = curve_fit(gaussian_distribution, x_coords, y_coords)
    mu, sigma, amp = popt[0], popt[1], popt[2]
    print('Mean: {} +\- {}'.format(mu, np.sqrt(pcov[0, 0])))
    print('Standard Deviation: {} +\- {}'.format(sigma, np.sqrt(pcov[1, 1])))
    print('Amplitude: {} +\- {}'.format(amp, np.sqrt(pcov[2, 2])))

    maximum_intensity = max(y_coords.tolist())
    print("maximum_intensity:", maximum_intensity)
    x_indices_of_max = np.argwhere(y_coords == np.amax(y_coords)).flatten().tolist()
    print("x_indices_of_max: ", x_indices_of_max)
    print("intensity at index = 0: ", y_coords[0])
    print("intensity at index = 170: ", y_coords[170])
    print("intensity at index = 172: ", y_coords[172])

    plt.axvline(mu + (1 * sigma), color='b', linestyle='dashed', linewidth=1, label="1sigma")
    plt.axvline(mu - (1 * sigma), color='b', linestyle='dashed', linewidth=1)
    plt.axvline(mu + (2 * sigma), color='r', linestyle='dashed', linewidth=1, label="2sigma")
    plt.axvline(mu - (2 * sigma), color='r', linestyle='dashed', linewidth=1)
    plt.axvline(mu + (3 * sigma), color='g', linestyle='dashed', linewidth=1, label="3sigma")
    plt.axvline(mu - (3 * sigma), color='g', linestyle='dashed', linewidth=1)
    plt.axvline(mu + (4 * sigma), color='gray', linestyle='dashed', linewidth=1, label="4sigma")
    plt.axvline(mu - (4 * sigma), color='gray', linestyle='dashed', linewidth=1)

    y_model_output = gaussian_distribution(x_coords, *popt)
    plt.plot(x_coords, y_coords, label='Data')
    plt.plot(x_coords, y_model_output, label='Model')
    plt.title("Scipy Optimize Fit - ./coregistration/cam_b_frame_186.png")
    plt.legend()
    # plt.show()
    fig.savefig("ScipyOptimizeCurveFit_For-coregistration-cam_b_frame_186.png")

    plt.close('all')


def get_noise_boundaries(image, coordinates_of_maxima, upper_limit_noise=10):
    # Steps of cropping surrounding noise
    print("Starting: get_noise_boundaries()")
    horizontal_center, vertical_center = coordinates_of_maxima[0], coordinates_of_maxima[1]
    horizontal_lineout = image[vertical_center, :]
    vertical_lineout = image[:, horizontal_center].flatten()
    first_i_above_uln = 0
    for i in range(len(horizontal_lineout)):
        x_index = i
        intensity = horizontal_lineout[i]
        if intensity > upper_limit_noise:
            print("At x = {}, y = {} ------ Intensity = {}".format(x_index, vertical_center, intensity))
            first_i_above_uln = x_index
            break

    first_j_above_uln = 0
    for j in range(len(vertical_lineout)):
        y_index = j
        intensity = vertical_lineout[j]
        if intensity > upper_limit_noise:
            print("At x = {}, y = {} ------ Intensity = {}".format(horizontal_center, y_index, intensity))
            first_j_above_uln = y_index
            break

    last_i_above_uln = 0
    for i in range(len(horizontal_lineout))[::-1]:  # , -1):
        x_index = i
        intensity = horizontal_lineout[i]
        # print(x_index, intensity)
        if intensity > upper_limit_noise:
            print("At x = {}, y = {} ------ Intensity = {}".format(x_index, vertical_center, intensity))
            last_i_above_uln = x_index
            break

    last_j_above_uln = 0
    for j in range(len(horizontal_lineout))[::-1]:  # , -1):
        y_index = j
        intensity = horizontal_lineout[j]
        # print(x_index, intensity)
        if intensity > upper_limit_noise:
            print("At x = {}, y = {} ------ Intensity = {}".format(horizontal_center, y_index, intensity))
            last_j_above_uln = y_index
            break

    index_of_last_pixel_above_noise_limit_flo_h = len(horizontal_lineout) - next(
        x[0] for x in enumerate(reversed(horizontal_lineout)) if x[1] > upper_limit_noise) - 1

    fig_a, ax = plt.subplots()
    # xdisplay, ydisplay = ax.transData.transform_point((horizontal_center, vertical_center))
    # print("xdisplay, ydisplay: ", xdisplay, ydisplay)
    ax.title.set_text('Image With NoiseSubtracted ROI Boxed')
    ax.axvline(first_i_above_uln, color='y', linestyle='dashed', linewidth=1)
    ax.axvline(last_i_above_uln, color='y', linestyle='dashed', linewidth=1)
    ax.axhline(first_j_above_uln, color='y', linestyle='dashed', linewidth=1)
    ax.axhline(last_j_above_uln, color='y', linestyle='dashed', linewidth=1)

    # ax.imshow(image)
    # fig_a.savefig("Image With NoiseSubtracted ROI Boxed.png")
    # plt.close('all')
    return first_i_above_uln, last_i_above_uln, first_j_above_uln, last_j_above_uln


def get_gaus_boundaries_x(image, coords_of_max):
    # fig = plt.figure()
    # xm, ym = coords_of_max
    ym, xm = coords_of_max
    # print("Inside get_gaus_boundaries_x(image, coords_of_max):")
    # print("coords_of_max = {}".format(coords_of_max))
    lineout = np.array(np.asarray(image[int(xm):int(xm) + 1, :])[0])
    indices = np.arange(0, len(lineout))
    # print("Maximum has a value of image[xm, ym], which is: ", image[xm, ym])

    try:
        p_initial = [image[int(xm), int(ym)] * 1.0, 960.00, 5.0, 0.0]
        popt, pcov = curve_fit(gauss, indices, lineout, p0=p_initial)

        amp, mu, sigma = popt[0], popt[1], popt[2]
        offset = popt[3]
        return int(mu), int(sigma), int(amp)
    except RuntimeError:
        raise cre.BeamNotGaussianException("The Beam lacks a Gaussian Profile (Horizontal)")


def get_gaus_boundaries_y(image, coords_of_max):
    image = np.array(image)

    ym, xm = coords_of_max

    p_initial = [image[xm, ym] * 1.0, 600.00, 5.0, 0.0]

    ys = np.array(image[:, ym:ym + 1].flatten())  # Vertical Line Out

    lineout = ys

    indices = np.arange(0, len(lineout))  # 0 to 1199

    try:
        popt, pcov = curve_fit(gauss, indices, lineout, p0=p_initial)

        amp, mu, sigma, offset = popt[0], popt[1], popt[2], popt[3]

        return mu, sigma, amp
    except RuntimeError:
        raise cre.BeamNotGaussianException("The Beam lacks a Gaussian Profile (Vertical)")


def save_img(filename, directory, image, sixteen_bit=True):
    """
    TODO: THIS IS THE THIRD SAVE_IMG, ARE WE EVEN USING IT!?
    """
    os.chdir(directory)
    if sixteen_bit:
        img = image.astype(np.uint16)
        img = np.asarray(img, dtype=np.uint16)
        img = Image.fromarray(img * 16)
        img.save(filename, compress_level=0)
    else:
        cv2.imwrite(filename, image.astype(np.uint16))
    os.chdir(directory)

