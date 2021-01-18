import traceback
from pypylon import genicam, pylon  # Import relevant pypylon packages/modules
import cv2
from image_processing import bit_depth_conversion as bdc
from image_processing import stack_images as stack
from coregistration import find_gaussian_profile as fgp
import numpy as np
import matplotlib.pyplot as plt
import sys
import numba
import csv as csv
import os
from PIL import Image, ImageDraw, ImageFont
from . import App as tk_app
from . import s1, s2, s3, s4, s5, s6
from . import s11
from path_management import image_management as im

y_n_msg = "Proceed? (y/n): "
sixteen_bit_max = (2 ** 16) - 1
twelve_bit_max = (2 ** 12) - 1
eight_bit_max = (2 ** 8) - 1

def progressBar(value, endvalue, bar_length=20):
    """
    Describe function

    :param value: Current progress value (i.e. the 57 in 57% out of 100_
    :param endvalue: Same example but the 100
    :param bar_length: Bar length in characters
    :return:
    """
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\rPercent: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()





EPSILON = sys.float_info.epsilon  # Smallest possible difference



@numba.njit
def hist1d_test(v, b, r):
    return np.histogram(v, b, r)


def set_xvalues(polygon, x0, x1):
    """
    Given a rectangular matplotlib.patches.Polygon object sets the horizontal values.

    Args:
        polygon: An instance of tlFactory.EnumerateDevices()
        x0: An integer
        x1: An integer
    Raises:
        Exception: TODO Add some error handling.

    """
    if len(polygon.get_xy()) == 4:
        _ndarray = polygon.get_xy()
        _ndarray[:, 0] = [x0, x0, x1, x1]
        polygon.set_xy(_ndarray)
    if len(polygon.get_xy()) == 5:
        _ndarray = polygon.get_xy()
        _ndarray[:, 0] = [x0, x0, x1, x1, x0]
        polygon.set_xy(_ndarray)

def add_histogram_representations(figure_a, figure_b, raw_array_a, raw_array_b):
    """
    Adds a matplotlib.pyplot.subplot to two matplotlib.pyplot.figure objects. The subplots are histograms of intensity
    data from raw_array_a and raw_array_b.
    Args:
        figure_a:
        figure_b:
        raw_array_a:
        raw_array_b:
    Returns:
        np.ndarray: An image array (3D [height, width, layers]) of the camera images and the corresponding histograms.
    """
    hist_img_a = np.fromstring(figure_a.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    hist_img_b = np.fromstring(figure_b.canvas.tostring_rgb(), dtype=np.uint8, sep='')  # convert  to image

    hist_img_a = hist_img_a.reshape(figure_a.canvas.get_width_height()[::-1] + (3,))
    hist_img_b = hist_img_b.reshape(figure_b.canvas.get_width_height()[::-1] + (3,))

    hist_width, hist_height = hist_img_a.shape[0], hist_img_a.shape[1]

    hist_img_a = cv2.cvtColor(hist_img_a, cv2.COLOR_RGB2BGR)  # img is rgb, convert to opencv's default bgr
    hist_img_b = cv2.cvtColor(hist_img_b, cv2.COLOR_RGB2BGR)  # img is rgb, convert to opencv's default bgr

    img_a_8bit_gray = bdc.to_8_bit(raw_array_a)
    img_b_8bit_gray = bdc.to_8_bit(raw_array_b)

    img_a_8bit_resized = cv2.cvtColor((stack.resize_img(img_a_8bit_gray, hist_width, hist_height)), cv2.COLOR_GRAY2BGR)
    img_b_8bit_resized = cv2.cvtColor((stack.resize_img(img_b_8bit_gray, hist_width, hist_height)), cv2.COLOR_GRAY2BGR)

    return np.vstack((np.hstack((hist_img_a, img_a_8bit_resized)), np.hstack((hist_img_b, img_b_8bit_resized))))

def initialize_histograms_rois(line_width=3):
    """
    Initializes histogram matplotlib.pyplot figures/subplots.

    Args:
        num_cameras: An integer
        line_width: An integer
    Raises:
        Exception: Any error/exception other than 'no such file or directory'.
    Returns:
        dict: A dictionary of histograms with ascending lowercase alphabetical letters (that match cameras) as keys
    """
    bins = 4096
    stream_subplots = dict()
    lines = {
        "intensities": dict(),
        "maxima": dict(),
        "averages": dict(),
        "stdevs": dict(),
        "max_vert": dict(),
        "avg+sigma": dict(),
        "avg-sigma": dict(),
        "grayscale_avg": dict(),
        "grayscale_avg+0.5sigma": dict(),
        "grayscale_avg-0.5sigma": dict()
    }
    fig_a = plt.figure(figsize=(5, 5))
    stream_subplots["a"] = fig_a.add_subplot()
    fig_b = plt.figure(figsize=(5, 5))
    stream_subplots["b"] = fig_b.add_subplot()



    for camera_identifier in ["a", "b"]:
        #camera_identifier = chr(97 + i)
        stream_subplots[camera_identifier].set_title('Camera ' + camera_identifier.capitalize())
        stream_subplots[camera_identifier].set_xlabel('Bin')
        stream_subplots[camera_identifier].set_ylabel('Frequency')

        lines["intensities"][camera_identifier], = stream_subplots[camera_identifier] \
            .plot(np.arange(bins), np.zeros((bins, 1)), c='k', lw=line_width, label='intensity')

        lines["maxima"][camera_identifier], = stream_subplots[camera_identifier] \
            .plot(np.arange(bins), np.zeros((bins, 1)), c='g', lw=1, label='maximum')

        lines["averages"][camera_identifier], = stream_subplots[camera_identifier] \
            .plot(np.arange(bins), np.zeros((bins, 1)), c='b', linestyle='dashed', lw=1, label='average')

        lines["stdevs"][camera_identifier], = stream_subplots[camera_identifier] \
            .plot(np.arange(bins), np.zeros((bins, 1)), c='r', linestyle='dotted', lw=2, label='stdev')

        lines["grayscale_avg"][camera_identifier] = stream_subplots[camera_identifier] \
            .axvline(-100, color='b', linestyle='dashed', linewidth=1)

        lines["grayscale_avg+0.5sigma"][camera_identifier] = stream_subplots[camera_identifier] \
            .axvline(-100, color='r', linestyle='dotted', linewidth=1)

        lines["grayscale_avg-0.5sigma"][camera_identifier] = stream_subplots[camera_identifier] \
            .axvline(-100, color='r', linestyle='dotted', linewidth=1)


        lines["max_vert"][camera_identifier] = stream_subplots[camera_identifier] \
            .axvline(-100, color='g', linestyle='solid', linewidth=1)

        lines["avg+sigma"][camera_identifier] = stream_subplots[camera_identifier] \
            .axvspan(-100, -100, alpha=0.5, color='#f5beba')

        lines["avg-sigma"][camera_identifier] = stream_subplots[camera_identifier] \
            .axvspan(-100, -100, alpha=0.5, color='#f5beba')

        stream_subplots[camera_identifier].set_xlim(-100, bins - 1 + 100)
        stream_subplots[camera_identifier].grid(True)
        stream_subplots[camera_identifier].set_autoscale_on(False)
        stream_subplots[camera_identifier].set_ylim(bottom=0, top=1)

    figs = dict()
    figs["a"], figs["b"] = fig_a, fig_b

    return figs, stream_subplots, lines



def initialize_histograms_algebra(line_width=3):
    """
    Initializes histogram matplotlib.pyplot figures/subplots.

    Args:
        num_cameras: An integer
        line_width: An integer
    Raises:
        Exception: Any error/exception other than 'no such file or directory'.
    Returns:
        dict: A dictionary of histograms with ascending lowercase alphabetical letters (that match cameras) as keys
    """
    bins = 4096
    stream_subplots = dict()
    lines = {
        "intensities": dict(),
        "maxima": dict(),
        "averages": dict(),
        "stdevs": dict(),
        "max_vert": dict(),
        "avg+sigma": dict(),
        "avg-sigma": dict(),
        "grayscale_avg": dict(),
        "grayscale_avg+0.5sigma": dict(),
        "grayscale_avg-0.5sigma": dict()
    }
    fig_a = plt.figure(figsize=(5, 5))
    stream_subplots["plus"] = fig_a.add_subplot()
    fig_b = plt.figure(figsize=(5, 5))
    stream_subplots["minus"] = fig_b.add_subplot()



    for camera_identifier in ["plus", "minus"]:
        #camera_identifier = chr(97 + i)
        stream_subplots[camera_identifier].set_xlabel('Bin')
        stream_subplots[camera_identifier].set_ylabel('Frequency')

        lines["intensities"][camera_identifier], = stream_subplots[camera_identifier] \
            .plot(np.arange(bins), np.zeros((bins, 1)), c='k', lw=line_width, label='intensity')

        lines["maxima"][camera_identifier], = stream_subplots[camera_identifier] \
            .plot(np.arange(bins), np.zeros((bins, 1)), c='g', lw=1, label='maximum')

        lines["averages"][camera_identifier], = stream_subplots[camera_identifier] \
            .plot(np.arange(bins), np.zeros((bins, 1)), c='b', linestyle='dashed', lw=1, label='average')

        lines["stdevs"][camera_identifier], = stream_subplots[camera_identifier] \
            .plot(np.arange(bins), np.zeros((bins, 1)), c='r', linestyle='dotted', lw=2, label='stdev')

        lines["grayscale_avg"][camera_identifier] = stream_subplots[camera_identifier] \
            .axvline(-100, color='b', linestyle='dashed', linewidth=1)

        lines["grayscale_avg+0.5sigma"][camera_identifier] = stream_subplots[camera_identifier] \
            .axvline(-100, color='r', linestyle='dotted', linewidth=1)

        lines["grayscale_avg-0.5sigma"][camera_identifier] = stream_subplots[camera_identifier] \
            .axvline(-100, color='r', linestyle='dotted', linewidth=1)

        lines["max_vert"][camera_identifier] = stream_subplots[camera_identifier] \
            .axvline(-100, color='g', linestyle='solid', linewidth=1)

        lines["avg+sigma"][camera_identifier] = stream_subplots[camera_identifier] \
            .axvspan(-100, -100, alpha=0.5, color='#f5beba')

        lines["avg-sigma"][camera_identifier] = stream_subplots[camera_identifier] \
            .axvspan(-100, -100, alpha=0.5, color='#f5beba')

        if camera_identifier is "plus":
            stream_subplots[camera_identifier].set_title("A Plus B Prime")
            stream_subplots["plus"].set_xlim(0, twelve_bit_max * 2 + 100)
        elif camera_identifier is "minus":
            stream_subplots[camera_identifier].set_title("A Minus B Prime")
            stream_subplots["minus"].set_xlim(-100 - twelve_bit_max, twelve_bit_max + 100)


        stream_subplots[camera_identifier].grid(True)
        stream_subplots[camera_identifier].set_autoscale_on(False)
        stream_subplots[camera_identifier].set_ylim(bottom=0, top=1)


    figs = dict()
    figs["plus"], figs["minus"] = fig_a, fig_b

    return figs, stream_subplots, lines



def initialize_histograms_r(line_width=3):
    """
    Initializes histogram matplotlib.pyplot figures/subplots.

    Args:
        num_cameras: An integer
        line_width: An integer
    Raises:
        Exception: Any error/exception other than 'no such file or directory'.
    Returns:
        dict: A dictionary of histograms with ascending lowercase alphabetical letters (that match cameras) as keys
    """
    bins = 4096
    stream_subplots = dict()
    lines = {
        "intensities": dict(),
        "maxima": dict(),
        "averages": dict(),
        "stdevs": dict(),
        "max_vert": dict(),
        "avg+sigma": dict(),
        "avg-sigma": dict(),
        "grayscale_avg": dict(),
        "grayscale_avg+0.5sigma": dict(),
        "grayscale_avg-0.5sigma": dict()
    }
    fig_a = plt.figure(figsize=(5, 5))
    stream_subplots["r"] = fig_a.add_subplot()


    for camera_identifier in ["r"]:
        #camera_identifier = chr(97 + i)
        stream_subplots[camera_identifier].set_xlabel('Bin')
        stream_subplots[camera_identifier].set_ylabel('Frequency')

        lines["intensities"][camera_identifier], = stream_subplots[camera_identifier] \
            .plot(np.arange(bins), np.zeros((bins, 1)), c='k', lw=line_width, label='intensity')

        lines["maxima"][camera_identifier], = stream_subplots[camera_identifier] \
            .plot(np.arange(bins), np.zeros((bins, 1)), c='g', lw=1, label='maximum')

        lines["averages"][camera_identifier], = stream_subplots[camera_identifier] \
            .plot(np.arange(bins), np.zeros((bins, 1)), c='b', linestyle='dashed', lw=1, label='average')

        lines["stdevs"][camera_identifier], = stream_subplots[camera_identifier] \
            .plot(np.arange(bins), np.zeros((bins, 1)), c='r', linestyle='dotted', lw=2, label='stdev')

        lines["grayscale_avg"][camera_identifier] = stream_subplots[camera_identifier] \
            .axvline(-100, color='b', linestyle='dashed', linewidth=1)

        lines["grayscale_avg+0.5sigma"][camera_identifier] = stream_subplots[camera_identifier] \
            .axvline(-100, color='r', linestyle='dotted', linewidth=1)

        lines["grayscale_avg-0.5sigma"][camera_identifier] = stream_subplots[camera_identifier] \
            .axvline(-100, color='r', linestyle='dotted', linewidth=1)


        lines["max_vert"][camera_identifier] = stream_subplots[camera_identifier] \
            .axvline(-100, color='g', linestyle='solid', linewidth=1)

        lines["avg+sigma"][camera_identifier] = stream_subplots[camera_identifier] \
            .axvspan(-100, -100, alpha=0.5, color='#f5beba')

        lines["avg-sigma"][camera_identifier] = stream_subplots[camera_identifier] \
            .axvspan(-100, -100, alpha=0.5, color='#f5beba')

        stream_subplots[camera_identifier].set_title("R Matrix Histogram")
        stream_subplots[camera_identifier].set_xlim(-1.2, 1.2)

        stream_subplots[camera_identifier].grid(True)
        stream_subplots[camera_identifier].set_autoscale_on(False)
        stream_subplots[camera_identifier].set_ylim(bottom=0, top=1)




    figs = dict()
    figs["r"] = fig_a

    return figs, stream_subplots, lines




def update_histogram(histogram_dict, lines_dict, identifier, bins, raw_2d_array,threshold=1.2, plus=False, minus=False, r=False):
    """
    Updates histograms for a given camera given the histogram of intensity values.

    Args:
        histogram_dict: TODO Add Description
        lines_dict: TODO Add Description
        identifier: TODO Add Description
        bins: TODO Add Description
        raw_2d_array: TODO Add Description
        threshold: TODO Add Description
    Raises:
        Exception: TODO Add Description
    Returns:
        TODO Add Description
    """
    if not plus and not minus and not r:
        calculated_hist = cv2.calcHist([raw_2d_array], [0], None, [bins], [0, 4095]) / np.prod(raw_2d_array.shape[:2])
        histogram_maximum = np.amax(calculated_hist)
        greyscale_max = np.amax(raw_2d_array.flatten())
        greyscale_avg = np.mean(raw_2d_array)
        greyscale_stdev = np.std(raw_2d_array)

        lines_dict["intensities"][identifier].set_ydata(calculated_hist)  # Intensities/Percent of Saturation

        lines_dict["maxima"][identifier].set_ydata(greyscale_max)  # Maximums
        lines_dict["averages"][identifier].set_ydata(greyscale_avg)  # Averages
        lines_dict["stdevs"][identifier].set_ydata(greyscale_stdev)  # Standard Deviations
        lines_dict["max_vert"][identifier].set_xdata(greyscale_max)  # Maximum Indicator as vertical line
        lines_dict["grayscale_avg"][identifier].set_xdata(greyscale_avg)  # Maximum Indicator as vertical line
        lines_dict["grayscale_avg+0.5sigma"][identifier].set_xdata(min([bins, greyscale_avg+(greyscale_stdev*0.5)]))  # Maximum Indicator as vertical line
        lines_dict["grayscale_avg-0.5sigma"][identifier].set_xdata(max([greyscale_avg-(greyscale_stdev*0.5), 0]))  # Maximum Indicator as vertical line

        set_xvalues(lines_dict["avg+sigma"][identifier], greyscale_avg, min([bins, greyscale_avg+(greyscale_stdev*0.5)]))
        set_xvalues(lines_dict["avg-sigma"][identifier], max([greyscale_avg-(greyscale_stdev*0.5), 0]), greyscale_avg)

        histogram_dict[identifier].legend(
            labels=(
                "intensity",
                "maximum %.0f" % greyscale_max,
                "average %.2f" % greyscale_avg,
                "stdev %.4f" % greyscale_stdev,),
            loc="upper right"
        )

        if histogram_maximum > 0.001:
            histogram_dict[identifier].set_ylim(bottom=0.000000, top=histogram_maximum * threshold)
        else:
            histogram_dict[identifier].set_ylim(bottom=0.000000, top=0.001)
    elif plus and not minus:
        plus_bins_ = (0, twelve_bit_max * 2)
        plus_bins_ = np.asarray(plus_bins_).astype(np.int64)
        ranges = (0, twelve_bit_max * 2)
        ranges = np.asarray(ranges).astype(np.float64)
        hist_output = hist1d_test(raw_2d_array.flatten(), twelve_bit_max * 2, (ranges[0], ranges[1]-1))
        h = hist_output[0]
        x_vals = hist_output[1][:-1]

        calculated_hist = h/ np.prod(raw_2d_array.shape[:2])
        histogram_maximum = np.amax(calculated_hist)
        greyscale_max = np.amax(raw_2d_array.flatten())
        greyscale_avg = np.mean(raw_2d_array)
        greyscale_stdev = np.std(raw_2d_array)

        lines_dict["intensities"][identifier].set_data(x_vals ,calculated_hist)  # Intensities/Percent of Saturation

        lines_dict["maxima"][identifier].set_ydata(greyscale_max)  # Maximums
        lines_dict["averages"][identifier].set_ydata(greyscale_avg)  # Averages
        lines_dict["stdevs"][identifier].set_ydata(greyscale_stdev)  # Standard Deviations
        lines_dict["max_vert"][identifier].set_xdata(greyscale_max)  # Maximum Indicator as vertical line
        lines_dict["grayscale_avg"][identifier].set_xdata(greyscale_avg)  # Maximum Indicator as vertical line
        lines_dict["grayscale_avg+0.5sigma"][identifier].set_xdata(min([bins, greyscale_avg+(greyscale_stdev*0.5)]))  # Maximum Indicator as vertical line
        lines_dict["grayscale_avg-0.5sigma"][identifier].set_xdata(max([greyscale_avg-(greyscale_stdev*0.5), 0]))  # Maximum Indicator as vertical line

        set_xvalues(lines_dict["avg+sigma"][identifier], greyscale_avg, min([bins, greyscale_avg+(greyscale_stdev*0.5)]))
        set_xvalues(lines_dict["avg-sigma"][identifier], max([greyscale_avg-(greyscale_stdev*0.5), 0]), greyscale_avg)

        histogram_dict[identifier].legend(
            labels=(
                "intensity",
                "maximum %.0f" % greyscale_max,
                "average %.2f" % greyscale_avg,
                "stdev %.4f" % greyscale_stdev,),
            loc="upper right"
        )

        if histogram_maximum > 0.001:
            histogram_dict[identifier].set_ylim(bottom=0.000000, top=histogram_maximum * threshold)
        else:
            histogram_dict[identifier].set_ylim(bottom=0.000000, top=0.001)
    elif minus and not plus:
        plus_bins_ = (0, twelve_bit_max * 2)
        plus_bins_ = np.asarray(plus_bins_).astype(np.int64)
        ranges = (-1*twelve_bit_max, twelve_bit_max)
        ranges = np.asarray(ranges).astype(np.float64)
        hist_output = hist1d_test(raw_2d_array.flatten(), twelve_bit_max * 2, (ranges[0], ranges[1]-1))
        h = hist_output[0]
        x_vals = hist_output[1][:-1]

        calculated_hist = h/ np.prod(raw_2d_array.shape[:2])
        histogram_maximum = np.amax(calculated_hist)
        greyscale_max = np.amax(raw_2d_array.flatten())
        greyscale_avg = np.mean(raw_2d_array)
        greyscale_stdev = np.std(raw_2d_array)

        lines_dict["intensities"][identifier].set_data(x_vals, calculated_hist)  # Intensities/Percent of Saturation

        lines_dict["maxima"][identifier].set_ydata(greyscale_max)  # Maximums
        lines_dict["averages"][identifier].set_ydata(greyscale_avg)  # Averages
        lines_dict["stdevs"][identifier].set_ydata(greyscale_stdev)  # Standard Deviations
        lines_dict["max_vert"][identifier].set_xdata(greyscale_max)  # Maximum Indicator as vertical line
        lines_dict["grayscale_avg"][identifier].set_xdata(greyscale_avg)  # Maximum Indicator as vertical line
        lines_dict["grayscale_avg+0.5sigma"][identifier].set_xdata(min([bins, greyscale_avg+(greyscale_stdev*0.5)]))  # Maximum Indicator as vertical line
        lines_dict["grayscale_avg-0.5sigma"][identifier].set_xdata(max([greyscale_avg-(greyscale_stdev*0.5), 0]))  # Maximum Indicator as vertical line

        set_xvalues(lines_dict["avg+sigma"][identifier], greyscale_avg, min([bins, greyscale_avg+(greyscale_stdev*0.5)]))
        set_xvalues(lines_dict["avg-sigma"][identifier], max([greyscale_avg-(greyscale_stdev*0.5), 0]), greyscale_avg)

        histogram_dict[identifier].legend(
            labels=(
                "intensity",
                "maximum %.0f" % greyscale_max,
                "average %.2f" % greyscale_avg,
                "stdev %.4f" % greyscale_stdev,),
            loc="upper right"
        )

        if histogram_maximum > 0.001:
            histogram_dict[identifier].set_ylim(bottom=0.000000, top=histogram_maximum * threshold)
        else:
            histogram_dict[identifier].set_ylim(bottom=0.000000, top=0.001)
    if r and (not plus and not minus):
        r_bins_ = (0, 1)
        r_bins_ = np.asarray(r_bins_).astype(np.int64)
        ranges = (float(-1), float(1))
        ranges = np.asarray(ranges).astype(np.float64)
        hist_output = hist1d_test(raw_2d_array.flatten(), 2000, (ranges[0], ranges[1]))
        h = hist_output[0]
        x_vals = hist_output[1][:-1]

        calculated_hist = h/ np.prod(raw_2d_array.shape[:2])
        histogram_maximum = np.amax(calculated_hist)
        greyscale_max = np.amax(raw_2d_array.flatten())
        greyscale_avg = np.mean(raw_2d_array)
        greyscale_stdev = np.std(raw_2d_array)

        lines_dict["intensities"][identifier].set_data(x_vals, calculated_hist)  # Intensities/Percent of Saturation

        lines_dict["maxima"][identifier].set_ydata(greyscale_max)  # Maximums
        lines_dict["averages"][identifier].set_ydata(greyscale_avg)  # Averages
        lines_dict["stdevs"][identifier].set_ydata(greyscale_stdev)  # Standard Deviations
        lines_dict["max_vert"][identifier].set_xdata(greyscale_max)  # Maximum Indicator as vertical line
        lines_dict["grayscale_avg"][identifier].set_xdata(greyscale_avg)  # Maximum Indicator as vertical line
        lines_dict["grayscale_avg+0.5sigma"][identifier].set_xdata(min([bins, greyscale_avg+(greyscale_stdev*0.5)]))  # Maximum Indicator as vertical line
        lines_dict["grayscale_avg-0.5sigma"][identifier].set_xdata(max([greyscale_avg-(greyscale_stdev*0.5), 0]))  # Maximum Indicator as vertical line

        set_xvalues(lines_dict["avg+sigma"][identifier], greyscale_avg, min([bins, greyscale_avg+(greyscale_stdev*0.5)]))
        set_xvalues(lines_dict["avg-sigma"][identifier], max([greyscale_avg-(greyscale_stdev*0.5), 0]), greyscale_avg)

        histogram_dict[identifier].legend(
            labels=(
                "intensity",
                "maximum %.0f" % greyscale_max,
                "average %.2f" % greyscale_avg,
                "stdev %.4f" % greyscale_stdev,),
            loc="upper right"
        )

        if histogram_maximum > 0.001:
            histogram_dict[identifier].set_ylim(bottom=0.000000, top=histogram_maximum * threshold)
        else:
            histogram_dict[identifier].set_ylim(bottom=0.000000, top=0.001)




class Stream:
    def __init__(self, fb=-1, save_imgs=False):
        self.save_imgs = save_imgs
        self.a_frames = list()
        self.b_frames = list()
        self.b_prime_frames = list()
        self.r_frames = list()
        self.r_frames_as_csv_text = list()

        self.cam_a = None
        self.cam_b = None
        self.all_cams = None
        self.latest_grab_results = {"a": None, "b": None}
        self.frame_count = 0
        self.frame_break = fb
        self.break_key = 'q'
        self.coregistration_break_key = 'c'  # Irrelevant
        self.keypoints_break_key = 'k'       # Irrelevant
        self.current_frame_a = None
        self.current_frame_b = None
        self.histocam_a = None
        self.histocam_b = None
        self.stacked_streams = None
        self.data_directory = None

        self.mu_x, self.sigma_x_a, self.amp_x = None, None, None
        self.mu_y, self.sigma_y_a, self.amp_y = None, None, None
        self.mu_x, self.sigma_x_b, self.amp_x = None, None, None
        self.mu_y, self.sigma_y_b, self.amp_y = None, None, None

        self.sigma_a_x, self.sigma_b_x = None, None
        self.sigma_a_y, self.sigma_b_y = None, None

        self.static_center_a = None
        self.static_center_b = None

        self.static_sigmas_x = None
        self.static_sigmas_y = None

        self.roi_a = None
        self.roi_b = None

        self.current_run = None


        self.warp_matrix = None
        self.warp_matrix_2 = None

        self.jump_level = 0


    def get_12bit_a_frames(self):
        return self.a_frames

    def get_12bit_b_frames(self):
        return self.b_frames

    def get_max_sigmas(self, guas_params_a_x, guas_params_a_y, guas_params_b_x, guas_params_b_y):
        mu_a_x, sigma_a_x, amp_a_x = guas_params_a_x
        mu_a_y, sigma_a_y, amp_a_y = guas_params_a_y

        mu_b_x, sigma_b_x, amp_b__x = guas_params_b_x
        mu_b_y, sigma_b_y, amp_b_y = guas_params_b_y

        max_sigma_x = max(sigma_a_x, sigma_b_x)
        max_sigma_y = max(sigma_a_y, sigma_b_y)

        return max_sigma_x, max_sigma_y

    def get_static_sigmas(self):
        return self.static_sigmas_x, self.static_sigmas_y

    def get_static_centers(self):
        return self.static_center_a, self.static_center_b

    def get_warp_matrix(self):
        return self.warp_matrix

    def get_warp_matrix2(self):
        return self.warp_matrix_2

    def set_current_run(self, datetime_string):
        self.current_run = datetime_string

    def set_static_sigmas(self, x, y):
        self.static_sigmas_x, self.static_sigmas_y = x, y

    def set_static_centers(self, a, b):
        self.static_center_a, self.static_center_b = a, b

    def set_warp_matrix(self, w):
        self.warp_matrix = w

    def set_warp_matrix2(self, w):
        self.warp_matrix_2 = w

    def set_warp_matrix2(self, w):
        self.warp_matrix_2 = w


    def offer_to_jump(self):
        offer = input("Would you like to use the previous parameters to JUMP to a specific step? (y/n): ")
        if offer.lower() == 'y':
            print("Step 1       : Stream Raw Camera Feed")
            print("Step 2       : Co-Register with Euclidean Transform")
            print("Step 3       : Find Brightest Pixel Locations")
            print("Step 4       : Set Gaussian-Based Static Centers")
            print("Step 5       : Define Regions of Interest ")
            print("Step 6A - 6C : Close in on ROI & Re-Co Register")
            print("Step 7       : Commence Image Algebra (Free Stream)")
            jump_level_input = int(input("Which level would you like to jump to?  "))
            self.jump_level = jump_level_input

    def get_cameras(self, config_files):
        """
        Should be called AFTER and with the return value of find_devices() (as implied by the first parameter: devices)
        Args:
            config_files: An integer
        Raises:
            Exception: Any error/exception other than 'no such file or directory'.
        Returns:
            dict: A dictionary of cameras with ascending lowercase alphabetical letters as keys.
        """

        tlFactory = pylon.TlFactory.GetInstance()  # Get the transport layer factory.
        devices = tlFactory.EnumerateDevices()  # Get all attached devices and exit application if no device is found.

        cameras = dict()

        # Create an array of instant cameras for the found devices and avoid exceeding a maximum number of devices.
        instant_camera_array = pylon.InstantCameraArray(min(len(devices), 2))
        self.all_cams = instant_camera_array

        for i, cam in enumerate(instant_camera_array):
            cam.Attach(tlFactory.CreateDevice(devices[i]))
            print("Camera ", i, "- Using device ", cam.GetDeviceInfo().GetModelName())  # Print camera model number

            cam.Open()
            # 1st camera will be a (ASCII = 97 + 0 = 97), 2nd will be b (ASCII = 97 + 1 = 98) and so on.
            pylon.FeaturePersistence.Load(config_files[chr(97 + i)], cam.GetNodeMap())
            cameras[chr(97 + i)] = cam

            if i == 0:
                self.cam_a = cam
            if i == 1:
                self.cam_b = cam

        self.all_cams = instant_camera_array

    def keep_streaming(self):
        if not self.all_cams.IsGrabbing():
            return False
        if self.frame_count == self.frame_break:
            return False
        if cv2.waitKey(1) & 0xFF == ord(self.break_key):
            return False
        return True

    def find_centers(self, frame_a_16bit, frame_b_16bit):

        x_a, y_a = fgp.get_coordinates_of_maximum(frame_a_16bit)
        x_b, y_b = fgp.get_coordinates_of_maximum(frame_b_16bit)

        return (x_a, y_a), (x_b, y_b)


    def grab_frames(self, warp_matrix=None, warp_matrix2=None):
        try:
            grab_result_a = self.cam_a.RetrieveResult(6000000, pylon.TimeoutHandling_ThrowException)
            grab_result_b = self.cam_b.RetrieveResult(6000000, pylon.TimeoutHandling_ThrowException)
            if grab_result_a.GrabSucceeded() and grab_result_b.GrabSucceeded():
                a, b = grab_result_a.GetArray(), grab_result_b.GetArray()
                if self.save_imgs:
                    self.a_frames.append(a)
                    self.b_frames.append(b)

                if warp_matrix is None:
                    return a, b
                else:
                    b1_shape = b.shape[1], b.shape[0]
                    b_prime = cv2.warpAffine(b, warp_matrix, b1_shape, flags=cv2.WARP_INVERSE_MAP)
                    self.b_prime_frames.append(b_prime)
                    return a, b_prime
        except Exception as e:
            traceback.print_exc()
            raise e

    def grab_frames2(self, roi_a, roi_b, warp_matrix_2):
        if warp_matrix_2 is None:
            return roi_a, roi_b

        roi_shape = roi_b.shape[1], roi_b.shape[0]
        roi_b_double_prime = cv2.warpAffine(roi_b, warp_matrix_2, roi_shape, flags=cv2.WARP_INVERSE_MAP)
        return roi_a, roi_b_double_prime



    def show_16bit_representations(self, a_as_12bit, b_as_12bit, b_prime=False, show_centers=False):
        a_as_16bit = bdc.to_16_bit(a_as_12bit)
        b_as_16bit = bdc.to_16_bit(b_as_12bit)
        if not show_centers:
            if not b_prime:
                cv2.imshow("Cam A", a_as_16bit)
                cv2.imshow("Cam B", b_as_16bit)
            else:
                cv2.imshow("A", a_as_16bit)
                cv2.imshow("B Prime", b_as_16bit)
        else:
            center_a, center_b = self.find_centers(a_as_16bit, b_as_16bit)
            a, b = self.imgs_w_centers(a_as_16bit, center_a, b_as_16bit, center_b)
            if not b_prime:
                cv2.imshow("Cam A", a)
                cv2.imshow("Cam B", b)
            else:
                cv2.imshow("A", a)
                cv2.imshow("B Prime", b)


    def imgs_w_centers(self, a_16bit_color, center_a, b_16bit_color, center_b):
        img_a = cv2.circle(a_16bit_color, center_a, 10, (0, eight_bit_max, 0), 2)
        img_b = cv2.circle(b_16bit_color, center_b, 10, (0, eight_bit_max, 0), 2)
        return img_a, img_b

    def full_img_w_roi_borders(self, img_12bit, center_):

        try:
            mu_x, sigma_x, amp_x = fgp.get_gaus_boundaries_x(img_12bit, center_)
            mu_y, sigma_y, amp_y = fgp.get_gaus_boundaries_y(img_12bit, center_)
            center_x,  center_y = int(center_[0]), int(center_[1])


            try:

                img_12bit[:, int(center_[0]) + int(sigma_x * 2)] = twelve_bit_max
                img_12bit[:, int(center_[0]) - int(sigma_x * 2)] = twelve_bit_max

                img_12bit[int(center_[1]) + int(sigma_y * 2), :] = twelve_bit_max
                img_12bit[int(center_[1]) - int(sigma_y * 2), :] = twelve_bit_max

            except IndexError:
                print("Warning: 4 sigma > frame height or width.")

        except RuntimeError:
            print("Warning: RuntimeError occurred while trying to calculate gaussian! ")

        return img_12bit

    def pre_alignment(self, histogram=False, centers=False, roi_borders=False, crop=False):
        a, b = self.current_frame_a, self.current_frame_b

        if roi_borders:
            a_as_16bit = bdc.to_16_bit(a)
            b_as_16bit = bdc.to_16_bit(b)

            if self.static_center_a is None or self.static_center_b is None:
                ca, cb = self.find_centers(a_as_16bit, b_as_16bit)
                a = self.full_img_w_roi_borders(a, ca)
                b = self.full_img_w_roi_borders(b, cb)
            else:
                print("Cam A:")
                a = self.full_img_w_roi_borders(a, self.static_center_a)
                print("Cam B:")
                b = self.full_img_w_roi_borders(b, self.static_center_b)


        if histogram:
            self.histocam_a.update(a)
            self.histocam_b.update(b)
            histocams = add_histogram_representations(self.histocam_a.get_figure(), self.histocam_b.get_figure(), a, b)
            cv2.imshow("Cameras with Histograms", histocams)
        else:
            if roi_borders or crop:
                self.show_16bit_representations(a, b, False, False)
            else:
                self.show_16bit_representations(a, b, False, centers)

    def start(self, histogram=False):
        continue_stream = False
        self.all_cams.StartGrabbing()

        step = 1
        if self.jump_level <= step:
            start = input("Step 1 - Stream Raw Camera Feed -  {}".format(y_n_msg)).lower()
            display_stream = True if start == "y" else False
            s1.step_one(self, histogram, display_stream, continue_stream)

        step = 2
        if self.jump_level <= step:
            s2.step_two(self, continue_stream)

        cv2.destroyAllWindows()

        step = 3
        if self.jump_level <= step:

            find_centers_ = input("Step 3 - Find Brightest Pixel Locations - {}".format(y_n_msg))
            s3.step_three(self, continue_stream, find_centers_)

        if self.warp_matrix is None:
            self.jump_level = 10

        step = 4
        if self.jump_level <= step:
            s4.step_four(self)

        cv2.destroyAllWindows()

        step = 5
        if self.jump_level <= step:
            s5.step_five(self, continue_stream)


        app = tk_app.App()
        step = 6

        if self.jump_level <= step:
            s6.step_six_a(self, continue_stream)
            s6.step_six_b(self, continue_stream, app)
            s6.step_six_c(self, continue_stream)

        cv2.destroyAllWindows()

        figs, histograms, lines = initialize_histograms_rois()
        figs_alg, histograms_alg, lines_alg = initialize_histograms_algebra()
        figs_r, histograms_r, lines_r = initialize_histograms_r()


        step = 7

        if self.static_center_a is None or self.static_center_b is None:
            print("Regions of Interest not defined: Exiting Program")
            continue_stream = False

        if self.jump_level == step:
            continue_stream = True
        elif self.jump_level > step:
            continue_stream = False
        else:
            start_algebra = input("Step 7 - Commence Image Algebra (Free Stream): Proceed? (y/n): ")
            if start_algebra.lower() == "y":
                continue_stream = True

        #root.mainloop()

        while continue_stream:
            self.frame_count += 1
            self.current_frame_a, self.current_frame_b = self.grab_frames(warp_matrix=self.warp_matrix)


            x_a, y_a = self.static_center_a
            x_b, y_b = self.static_center_b

            n_sigma = 3


            self.roi_a = self.current_frame_a[
                         y_a - n_sigma * self.static_sigmas_y: y_a + n_sigma * self.static_sigmas_y + n_sigma,
                         x_a - n_sigma*self.static_sigmas_x: x_a + n_sigma*self.static_sigmas_x + n_sigma]

            self.roi_b = self.current_frame_b[
                         y_b - n_sigma * self.static_sigmas_y: y_b + n_sigma * self.static_sigmas_y + n_sigma,
                         x_b - n_sigma * self.static_sigmas_x: x_b + n_sigma*self.static_sigmas_x + n_sigma]

            if self.warp_matrix_2 is None:
                roi_a = self.roi_a
                b_double_prime = self.roi_b
            else:
                roi_a, b_double_prime = self.grab_frames2(self.roi_a.copy(), self.roi_b.copy(), self.warp_matrix_2.copy())


            CENTER_B_DP = int(b_double_prime.shape[1]*0.5), int(b_double_prime.shape[0]*0.5)


            x_a, y_a = CENTER_B_DP
            x_b, y_b = CENTER_B_DP
            n_sigma = app.foo



            self.roi_a = self.roi_a[
                         int(y_a - n_sigma * self.static_sigmas_y): int(y_a + n_sigma * self.static_sigmas_y + 1),
                         int(x_a - n_sigma * self.static_sigmas_x): int(x_a + n_sigma * self.static_sigmas_x + 1)]

            b_double_prime = b_double_prime[
                         int(y_b - n_sigma * self.static_sigmas_y): int(y_b + n_sigma * self.static_sigmas_y + 1),
                         int(x_b - n_sigma * self.static_sigmas_x): int(x_b + n_sigma * self.static_sigmas_x + 1)]

            self.roi_b = b_double_prime
            h = b_double_prime.shape[0]
            w = b_double_prime.shape[1]



            update_histogram(histograms, lines, "a", 4096, self.roi_a)
            update_histogram(histograms, lines, "b", 4096, self.roi_b)
            figs["a"].canvas.draw()  # Draw updates subplots in interactive mode
            figs["b"].canvas.draw()  # Draw updates subplots in interactive mode
            hist_img_a = np.fromstring(figs["a"].canvas.tostring_rgb(), dtype=np.uint8, sep='')
            hist_img_b = np.fromstring(figs["b"].canvas.tostring_rgb(), dtype=np.uint8, sep='')  # convert  to image
            hist_img_a = hist_img_a.reshape(figs["a"].canvas.get_width_height()[::-1] + (3,))
            hist_img_b = hist_img_b.reshape(figs["b"].canvas.get_width_height()[::-1] + (3,))
            hist_img_a = cv2.resize(hist_img_a, (w, h), interpolation=cv2.INTER_AREA)
            hist_img_b = cv2.resize(hist_img_b, (w, h), interpolation=cv2.INTER_AREA)
            hist_img_a = bdc.to_16_bit(cv2.resize(hist_img_a, (w, h), interpolation=cv2.INTER_AREA), 8)
            hist_img_b = bdc.to_16_bit(cv2.resize(hist_img_b, (w, h), interpolation=cv2.INTER_AREA), 8)

            ROI_A_WITH_HISTOGRAM = np.concatenate((cv2.cvtColor(hist_img_a, cv2.COLOR_RGB2BGR), cv2.cvtColor(self.roi_a * 16, cv2.COLOR_GRAY2BGR)), axis=1)
            ROI_B_WITH_HISTOGRAM = np.concatenate((cv2.cvtColor(hist_img_b, cv2.COLOR_RGB2BGR), cv2.cvtColor(self.roi_b * 16, cv2.COLOR_GRAY2BGR)), axis=1)


            A_ON_B = np.concatenate((ROI_A_WITH_HISTOGRAM, ROI_B_WITH_HISTOGRAM), axis=0)

            plus_ = cv2.add(self.roi_a, self.roi_b)
            minus_ = np.zeros(self.roi_a.shape, dtype='int16')
            minus_ = np.add(minus_, self.roi_a)
            minus_ = np.add(minus_, self.roi_b * (-1))
            #print("Lowest pixel in the minus spectrum: {}".format(np.min(minus_.flatten())))

            update_histogram(histograms_alg, lines_alg, "plus", 4096, plus_, plus=True)
            update_histogram(histograms_alg, lines_alg, "minus", 4096, minus_, minus=True)


            displayable_plus = cv2.add(self.roi_a, self.roi_b) * 16
            displayable_minus = cv2.subtract(self.roi_a, self.roi_b) * 16

            figs_alg["plus"].canvas.draw()  # Draw updates subplots in interactive mode
            hist_img_plus = np.fromstring(figs_alg["plus"].canvas.tostring_rgb(), dtype=np.uint8, sep='')
            hist_img_plus = hist_img_plus.reshape(figs_alg["plus"].canvas.get_width_height()[::-1] + (3,))
            hist_img_plus = cv2.resize(hist_img_plus, (w, h), interpolation=cv2.INTER_AREA)
            hist_img_plus = bdc.to_16_bit(cv2.resize(hist_img_plus, (w, h), interpolation=cv2.INTER_AREA), 8)
            PLUS_WITH_HISTOGRAM = np.concatenate((cv2.cvtColor(hist_img_plus, cv2.COLOR_RGB2BGR), cv2.cvtColor(displayable_plus, cv2.COLOR_GRAY2BGR)), axis=1)

            figs_alg["minus"].canvas.draw()  # Draw updates subplots in interactive mode
            hist_img_minus = np.fromstring(figs_alg["minus"].canvas.tostring_rgb(), dtype=np.uint8, sep='')  # convert  to image
            hist_img_minus = hist_img_minus.reshape(figs_alg["minus"].canvas.get_width_height()[::-1] + (3,))
            hist_img_minus = cv2.resize(hist_img_minus, (w, h), interpolation=cv2.INTER_AREA)
            hist_img_minus = bdc.to_16_bit(cv2.resize(hist_img_minus, (w, h), interpolation=cv2.INTER_AREA), 8)
            MINUS_WITH_HISTOGRAM = np.concatenate((cv2.cvtColor(hist_img_minus, cv2.COLOR_RGB2BGR), cv2.cvtColor(displayable_minus, cv2.COLOR_GRAY2BGR)), axis=1)






            ALGEBRA = np.concatenate((PLUS_WITH_HISTOGRAM, MINUS_WITH_HISTOGRAM), axis=0)
            DASHBOARD = np.concatenate((A_ON_B, ALGEBRA), axis=1)
            dash_height, dash_width, dash_channels = DASHBOARD.shape
            if dash_width > 2000:
                scale_factor = float(float(2000)/float(dash_width))
                DASHBOARD = cv2.resize(DASHBOARD, (int(dash_width*scale_factor), int(dash_height*scale_factor)))
            cv2.imshow("Dashboard", DASHBOARD)


            R_MATRIX = np.divide(minus_, plus_)
            nan_mean = np.nanmean(R_MATRIX.flatten())
            nan_st_dev = np.nanstd(R_MATRIX.flatten())

            DISPLAYABLE_R_MATRIX = np.zeros((R_MATRIX.shape[0], R_MATRIX.shape[1], 3), dtype=np.uint8)
            DISPLAYABLE_R_MATRIX[:, :, 1] = np.where(R_MATRIX < 0.00, abs(R_MATRIX*(2**8 - 1)), 0)
            DISPLAYABLE_R_MATRIX[:, :, 2] = np.where(R_MATRIX < 0.00, abs(R_MATRIX*(2**8 - 1)), 0)

            DISPLAYABLE_R_MATRIX[:, :, 2] = np.where(R_MATRIX > 0.00, abs(R_MATRIX*(2**8 - 1)), DISPLAYABLE_R_MATRIX[:, :, 2])

            dr_height, dr_width, dr_channels = DISPLAYABLE_R_MATRIX.shape



            update_histogram(histograms_r, lines_r, "r", 4096, R_MATRIX, r=True)
            figs_r["r"].canvas.draw()  # Draw updates subplots in interactive mode
            hist_img_r = np.fromstring(figs_r["r"].canvas.tostring_rgb(), dtype=np.uint8, sep='')  # convert  to image
            hist_img_r = hist_img_r.reshape(figs_r["r"].canvas.get_width_height()[::-1] + (3,))
            hist_img_r = cv2.resize(hist_img_r, (w, h), interpolation=cv2.INTER_AREA)
            hist_img_r = bdc.to_16_bit(cv2.resize(hist_img_r, (w, h), interpolation=cv2.INTER_AREA), 8)
            R_HIST = (cv2.cvtColor(hist_img_r, cv2.COLOR_RGB2BGR))

            R_VALUES = Image.new('RGB', (dr_width, dr_height), (eight_bit_max, eight_bit_max, eight_bit_max))


            draw = ImageDraw.Draw(R_VALUES)
            font = ImageFont.truetype('arial.ttf', size=30)
            (x, y) = (50, 50)
            message = "R Matrix Values\n"
            message = message + "Average: {0:.4f}".format(nan_mean) + "\n"
            message = message + "Sigma: {0:.4f}".format(nan_st_dev)

            #Mean: {0:.4f}\n".format(nan_mean, 2.000*float(self.frame_count))
            color = 'rgb(0, 0, 0)'  # black color
            draw.text((x, y), message, fill=color, font=font)
            R_VALUES = np.array(R_VALUES)
            VALUES_W_HIST = np.concatenate((R_VALUES*(2**8), np.array(R_HIST)), axis=1)



            cv2.imshow("R_MATRIX", np.concatenate((VALUES_W_HIST, np.array(DISPLAYABLE_R_MATRIX*(2**8), dtype='uint16')), axis=1))


            continue_stream = self.keep_streaming()


            if not continue_stream:
                if app is not None:
                    app.callback()

                cv2.destroyAllWindows()

        satisfied_with_run = False


        current_r_frame = 0
        stats = list()
        a_frames = list()
        b_prime_frames = list()

        a_images = list()
        b_prime_images = list()

        start_writing_at = 0
        end_writing_at = 0


        step = 8
        if self.jump_level > 8:
            satisfied_with_run = True

        run_folder = os.path.join("D:", "\\" + self.current_run)

        while satisfied_with_run is False:

            current_r_frame = 0
            record_r_matrices = input("Step 8 - Image Algebra (Record): Proceed? (y/n): ")
            stats = list()
            self.r_frames = list()
            a_frames = list()
            b_prime_frames = list()

            a_images = list()
            b_prime_images = list()

            stats.append(["Frame", "Avg_R", "Sigma_R"])
            #r_matrix_limit = int(input("R Matrix Frame Break: "))
            if record_r_matrices.lower() == "y":
                continue_stream = True
                while continue_stream:
                    self.frame_count += 1
                    self.current_frame_a, self.current_frame_b = self.grab_frames(warp_matrix=self.warp_matrix)
                    current_r_frame += 1
                    print("Current R Frame: {}".format(current_r_frame))

                    x_a, y_a = self.static_center_a
                    x_b, y_b = self.static_center_b

                    n_sigma = 3

                    self.roi_a = self.current_frame_a[
                                 y_a - n_sigma * self.static_sigmas_y: y_a + n_sigma * self.static_sigmas_y + n_sigma,
                                 x_a - n_sigma * self.static_sigmas_x: x_a + n_sigma * self.static_sigmas_x + n_sigma]

                    self.roi_b = self.current_frame_b[
                                 y_b - n_sigma * self.static_sigmas_y: y_b + n_sigma * self.static_sigmas_y + n_sigma,
                                 x_b - n_sigma * self.static_sigmas_x: x_b + n_sigma * self.static_sigmas_x + n_sigma]

                    roi_a, b_double_prime = self.grab_frames2(self.roi_a.copy(), self.roi_b.copy(),
                                                              self.warp_matrix_2.copy())

                    CENTER_B_DP = int(b_double_prime.shape[1] * 0.5), int(b_double_prime.shape[0] * 0.5)

                    x_a, y_a = CENTER_B_DP
                    x_b, y_b = CENTER_B_DP
                    n_sigma = app.foo

                    a_frames.append(roi_a)
                    b_prime_frames.append(b_double_prime)
                    a_images.append(roi_a)
                    b_prime_images.append(b_double_prime)

                    self.roi_a = self.roi_a[
                                 int(y_a - n_sigma * self.static_sigmas_y): int(
                                     y_a + n_sigma * self.static_sigmas_y + 1),
                                 int(x_a - n_sigma * self.static_sigmas_x): int(
                                     x_a + n_sigma * self.static_sigmas_x + 1)]

                    b_double_prime = b_double_prime[
                                     int(y_b - n_sigma * self.static_sigmas_y): int(
                                         y_b + n_sigma * self.static_sigmas_y + 1),
                                     int(x_b - n_sigma * self.static_sigmas_x): int(
                                         x_b + n_sigma * self.static_sigmas_x + 1)]


                    self.roi_b = b_double_prime
                    h = b_double_prime.shape[0]
                    w = b_double_prime.shape[1]


                    update_histogram(histograms, lines, "a", 4096, self.roi_a)
                    update_histogram(histograms, lines, "b", 4096, self.roi_b)
                    figs["a"].canvas.draw()  # Draw updates subplots in interactive mode
                    figs["b"].canvas.draw()  # Draw updates subplots in interactive mode
                    hist_img_a = np.fromstring(figs["a"].canvas.tostring_rgb(), dtype=np.uint8, sep='')
                    hist_img_b = np.fromstring(figs["b"].canvas.tostring_rgb(), dtype=np.uint8, sep='')  # convert  to image
                    hist_img_a = hist_img_a.reshape(figs["a"].canvas.get_width_height()[::-1] + (3,))
                    hist_img_b = hist_img_b.reshape(figs["b"].canvas.get_width_height()[::-1] + (3,))
                    hist_img_a = cv2.resize(hist_img_a, (w, h), interpolation=cv2.INTER_AREA)
                    hist_img_b = cv2.resize(hist_img_b, (w, h), interpolation=cv2.INTER_AREA)
                    hist_img_a = bdc.to_16_bit(cv2.resize(hist_img_a, (w, h), interpolation=cv2.INTER_AREA), 8)
                    hist_img_b = bdc.to_16_bit(cv2.resize(hist_img_b, (w, h), interpolation=cv2.INTER_AREA), 8)

                    ROI_A_WITH_HISTOGRAM = np.concatenate(
                        (cv2.cvtColor(hist_img_a, cv2.COLOR_RGB2BGR), cv2.cvtColor(self.roi_a * 16, cv2.COLOR_GRAY2BGR)),
                        axis=1)
                    ROI_B_WITH_HISTOGRAM = np.concatenate(
                        (cv2.cvtColor(hist_img_b, cv2.COLOR_RGB2BGR), cv2.cvtColor(self.roi_b * 16, cv2.COLOR_GRAY2BGR)),
                        axis=1)

                    A_ON_B = np.concatenate((ROI_A_WITH_HISTOGRAM, ROI_B_WITH_HISTOGRAM), axis=0)

                    plus_ = cv2.add(self.roi_a, self.roi_b)
                    minus_ = np.zeros(self.roi_a.shape, dtype='int16')
                    minus_ = np.add(minus_, self.roi_a)
                    minus_ = np.add(minus_, self.roi_b * (-1))

                    update_histogram(histograms_alg, lines_alg, "plus", 4096, plus_, plus=True)
                    update_histogram(histograms_alg, lines_alg, "minus", 4096, minus_, minus=True)

                    displayable_plus = cv2.add(self.roi_a, self.roi_b) * 16
                    displayable_minus = cv2.subtract(self.roi_a, self.roi_b) * 16

                    figs_alg["plus"].canvas.draw()  # Draw updates subplots in interactive mode
                    hist_img_plus = np.fromstring(figs_alg["plus"].canvas.tostring_rgb(), dtype=np.uint8, sep='')
                    hist_img_plus = hist_img_plus.reshape(figs_alg["plus"].canvas.get_width_height()[::-1] + (3,))
                    hist_img_plus = cv2.resize(hist_img_plus, (w, h), interpolation=cv2.INTER_AREA)
                    hist_img_plus = bdc.to_16_bit(cv2.resize(hist_img_plus, (w, h), interpolation=cv2.INTER_AREA), 8)
                    PLUS_WITH_HISTOGRAM = np.concatenate((cv2.cvtColor(hist_img_plus, cv2.COLOR_RGB2BGR),
                                                          cv2.cvtColor(displayable_plus, cv2.COLOR_GRAY2BGR)), axis=1)

                    figs_alg["minus"].canvas.draw()  # Draw updates subplots in interactive mode
                    hist_img_minus = np.fromstring(figs_alg["minus"].canvas.tostring_rgb(), dtype=np.uint8,
                                                   sep='')  # convert  to image
                    hist_img_minus = hist_img_minus.reshape(figs_alg["minus"].canvas.get_width_height()[::-1] + (3,))
                    hist_img_minus = cv2.resize(hist_img_minus, (w, h), interpolation=cv2.INTER_AREA)
                    hist_img_minus = bdc.to_16_bit(cv2.resize(hist_img_minus, (w, h), interpolation=cv2.INTER_AREA), 8)
                    MINUS_WITH_HISTOGRAM = np.concatenate((cv2.cvtColor(hist_img_minus, cv2.COLOR_RGB2BGR),
                                                           cv2.cvtColor(displayable_minus, cv2.COLOR_GRAY2BGR)), axis=1)

                    ALGEBRA = np.concatenate((PLUS_WITH_HISTOGRAM, MINUS_WITH_HISTOGRAM), axis=0)
                    DASHBOARD = np.concatenate((A_ON_B, ALGEBRA), axis=1)
                    dash_height, dash_width, dash_channels = DASHBOARD.shape

                    if dash_width > 2000:
                        scale_factor = float(float(2000) / float(dash_width))
                        DASHBOARD = cv2.resize(DASHBOARD, (int(dash_width * scale_factor), int(dash_height * scale_factor)))

                    cv2.imshow("Dashboard", DASHBOARD)

                    R_MATRIX = np.divide(minus_, plus_)
                    self.r_frames.append(R_MATRIX)
                    nan_mean = np.nanmean(R_MATRIX.flatten())
                    nan_st_dev = np.nanstd(R_MATRIX.flatten())
                    stats.append([len(self.r_frames), nan_mean, nan_st_dev])



                    DISPLAYABLE_R_MATRIX = np.zeros((R_MATRIX.shape[0], R_MATRIX.shape[1], 3), dtype=np.uint8)
                    DISPLAYABLE_R_MATRIX[:, :, 1] = np.where(R_MATRIX < 0.00, abs(R_MATRIX * (2 ** 8 - 1)), 0)
                    DISPLAYABLE_R_MATRIX[:, :, 2] = np.where(R_MATRIX < 0.00, abs(R_MATRIX * (2 ** 8 - 1)), 0)

                    DISPLAYABLE_R_MATRIX[:, :, 2] = np.where(R_MATRIX > 0.00, abs(R_MATRIX * (2 ** 8 - 1)),
                                                             DISPLAYABLE_R_MATRIX[:, :, 2])

                    dr_height, dr_width, dr_channels = DISPLAYABLE_R_MATRIX.shape

                    update_histogram(histograms_r, lines_r, "r", 4096, R_MATRIX, r=True)
                    figs_r["r"].canvas.draw()  # Draw updates subplots in interactive mode
                    hist_img_r = np.fromstring(figs_r["r"].canvas.tostring_rgb(), dtype=np.uint8,
                                               sep='')  # convert  to image
                    hist_img_r = hist_img_r.reshape(figs_r["r"].canvas.get_width_height()[::-1] + (3,))
                    hist_img_r = cv2.resize(hist_img_r, (w, h), interpolation=cv2.INTER_AREA)
                    hist_img_r = bdc.to_16_bit(cv2.resize(hist_img_r, (w, h), interpolation=cv2.INTER_AREA), 8)

                    R_VALUES = Image.new('RGB', (dr_width, dr_height), (eight_bit_max, eight_bit_max, eight_bit_max))

                    # initialise the drawing context with
                    # the image object as background

                    draw = ImageDraw.Draw(R_VALUES)
                    font = ImageFont.truetype('arial.ttf', size=30)
                    (x, y) = (50, 50)
                    message = "R Matrix Values\n"
                    message = message + "Average: {0:.4f}".format(nan_mean) + "\n"
                    message = message + "Sigma: {0:.4f}".format(nan_st_dev)

                    # Mean: {0:.4f}\n".format(nan_mean, 2.000*float(self.frame_count))
                    color = 'rgb(0, 0, 0)'  # black color
                    draw.text((x, y), message, fill=color, font=font)
                    R_VALUES = np.array(R_VALUES)
                    VALUES_W_HIST = np.concatenate((R_VALUES * (2 ** 8), np.array(R_HIST)), axis=1)

                    cv2.imshow("R_MATRIX", np.concatenate(
                        (VALUES_W_HIST, np.array(DISPLAYABLE_R_MATRIX * (2 ** 8), dtype='uint16')), axis=1))



                    continue_stream = self.keep_streaming()
                    if continue_stream is False:
                        satisfied_with_range = False
                        while satisfied_with_range is False:
                            cv2.destroyAllWindows()
                            fig_ = plt.figure()
                            ax1 = fig_.add_subplot(111)
                            frames = list()
                            averages = list()
                            sigmas = list()

                            starting_frame = int(input("Start at frame: "))
                            end_frame = int(input("End at frame: "))

                            #for i in range(1, len(stats)):
                            for i in range(starting_frame, end_frame + 1):
                                frames.append(stats[i][0])
                                averages.append(stats[i][1])
                                sigmas.append(stats[i][2])

                            ax1.errorbar(frames, averages, yerr=sigmas, c='b', capsize=5)
                            ax1.set_xlabel('Frame')
                            ax1.set_ylabel('R (Mean)')
                            ax1.set_title('Mean R by Frame')
                            ax1.axhline(y=-1.0, xmin=starting_frame, xmax=end_frame)
                            ax1.axhline(y=0.0, xmin=starting_frame, xmax=end_frame)
                            ax1.axhline(y=1.0, xmin=starting_frame, xmax=end_frame)

                            save_path = os.path.join(run_folder, 'mean_r_by_frame.png')
                            fig_.savefig(save_path)
                            plot_img = cv2.imread(save_path, cv2.IMREAD_COLOR)
                            cv2.imshow('R Mean Plot', plot_img)
                            cv2.waitKey(60000)
                            cv2.destroyAllWindows()
                            range_satisfaction_input = input("Are you satisfied with this range? (y/n): ")
                            if range_satisfaction_input.lower() == "y":
                                satisfied_with_range = True
                                start_writing_at = starting_frame
                                end_writing_at = end_frame
                        satisfaction_input = input("Are you satisfied with this run? (y/n): ")
                        if satisfaction_input.lower() == 'y':
                            satisfied_with_run = True
            if record_r_matrices.lower() == "n":
                satisfied_with_run = True
                continue_stream = False

        cv2.destroyAllWindows()

        step = 9

        if not (self.jump_level > step):
            write_to_csv = input("Step 9 - Write Recorded R Frame(s) to File(s)? - Proceed? (y/n): ")
        else:
            write_to_csv = "n"

        if self.jump_level <= step and write_to_csv.lower() == 'y':

            r_matrices = self.r_frames
            n_ = 0
            print("Writing R Matrices")
            for i in range(start_writing_at, end_writing_at + 1):
                n_ += 1
                r_matrix = r_matrices[i - 1]
                csv_path = os.path.join(run_folder, "r_matrix_{}.csv".format(n_))
                with open(csv_path, "w+", newline='') as my_csv:
                    csvWriter = csv.writer(my_csv, delimiter=',')
                    csvWriter.writerows(r_matrix.tolist())

            n_ = 0
            a_frames_dir = os.path.join(run_folder, "cam_a_frames")
            print("Writing A Matrices")
            for i in range(start_writing_at, end_writing_at + 1):
            #for a_matrix in a_frames:
                n_ += 1
                a_matrix = a_frames[i - 1]
                csv_path = os.path.join(run_folder, "a_matrix_{}.csv".format(n_))
                with open(csv_path, "w+", newline='') as my_csv:
                    csvWriter = csv.writer(my_csv, delimiter=',')
                    csvWriter.writerows(a_matrix.tolist())

                #a16 = bdc.to_16_bit(a_matrix)
                #im.save_img("a_{}.png".format(n_), a_frames_dir, a16)

            print("Writing A Images")
            n_ = 0
            for i in range(start_writing_at, end_writing_at + 1):
            #for img_a in a_images:
                n_ += 1
                img_a = a_images[i-1]
                a16 = bdc.to_16_bit(img_a)
                im.save_img("a_{}.png".format(n_), a_frames_dir, a16)


            b_frames_dir = os.path.join(run_folder, "cam_b_frames")

            n_ = 0
            print("Writing B Matrices")
            for i in range(start_writing_at, end_writing_at + 1):
            #for b_matrix in b_prime_frames:
                n_ += 1
                b_matrix = b_prime_frames[i-1]
                csv_path = os.path.join(run_folder, "b_matrix_{}.csv".format(n_))
                with open(csv_path, "w+", newline='') as my_csv:
                    csvWriter = csv.writer(my_csv, delimiter=',')
                    csvWriter.writerows(b_matrix.tolist())

                #b16 = bdc.to_16_bit(b_matrix)
                #im.save_img("b_{}.png".format(n_), b_frames_dir, b16)

            print("Writing B Images")
            n_ = 0
            for i in range(start_writing_at, end_writing_at + 1):
            #for img_b in b_prime_images:
                n_ += 1
                img_b = b_prime_images[i-1]
                b16 = bdc.to_16_bit(img_b)
                im.save_img("b_{}.png".format(n_), b_frames_dir, b16)

            print("Writing R Matrix Stats to file:")
            stats_csv_path = os.path.join(run_folder, "r_matrices_stats.csv")
            with open(stats_csv_path, "w+", newline='') as stats_csv:
                stats_csvWriter = csv.writer(stats_csv, delimiter=',')
                stats_csvWriter.writerow(stats[0])
                count = 0
                for i in range(start_writing_at, end_writing_at + 1):
                    count += 1
                    stats_csvWriter.writerow([count, stats[i][1], stats[i][2]])

            print("Matrices and Matrix Stats have finished writing to file.")
        try:
            if app:
                if not app.at_front:
                    app.bring_to_front()
        except Exception:
            pass

        if app:
            app.callback()

        self.all_cams.StopGrabbing()

        step = 10
        notes = input("Step 10 - Write some notes to a file? - Proceed? (y/n): ")

        if notes.lower() == 'y':
            notes = input("Write notes below:\n\n")
            notes_file = open(os.path.join(run_folder, 'notes.txt'), 'w+')
            notes_file.write(notes)
            notes_file.close()

        step = 11
        s11.step_eleven(self, run_folder)



