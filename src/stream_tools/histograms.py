import numpy as np
import cv2
import numba

from image_processing import bit_depth_conversion as bdc
import matplotlib.pyplot as plt
from image_processing import stack_images as stack

y_n_msg = "Proceed? (y/n): "
sixteen_bit_max = (2 ** 16) - 1
twelve_bit_max = (2 ** 12) - 1
eight_bit_max = (2 ** 8) - 1


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

