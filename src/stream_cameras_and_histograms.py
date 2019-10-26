
"""How to grab and process images from multiple cameras using the CInstantCameraArray class. The CInstantCameraArray
class represents an array of instant camera objects. It provides almost the same interface as the instant camera for
grabbing. The main purpose of the CInstantCameraArray is to simplify waiting for images and camera events of multiple
cameras in one thread. This is done by providing a single RetrieveResult method for all cameras in the array.
Alternatively, the grabbing can be started using the internal grab loop threads of all cameras in the
CInstantCameraArray. The grabbed images can then be processed by one or more  image event handlers. """

# #from pypylon import pylon
# # import genicam
from pypylon import genicam, pylon  # Import relevant pypylon packages/modules
import matplotlib.pyplot as plt  # For plotting live histogram
# import os
# import sys
import cv2
import numpy as np  # Pixel math, among other array operations
import traceback # exception handling
# from numpy.linalg import inv
#
def find_devices():
    """
    If devices are connected to computer and number of cameras are connected , and number of devices connected is equal
    to or below user specified limit, this function should return list of Basler device instances. Otherwise, raises a
    a general Exception or a RuntimeError.

    Raises:
        Exception: Any error/exception other than 'no such file or directory'.
    """
    try:
        tlFactory = pylon.TlFactory.GetInstance()  # Get the transport layer factory.
        devices = tlFactory.EnumerateDevices()  # Get all attached devices and exit application if no device is found.
        return devices, tlFactory
    except Exception as e:  # Exception/Error handling
        print("An exception occurred:")
        traceback.print_exc()


def get_cameras(devices, tlFactory, config_files, num_cameras=2):
    """
    Should be called AFTER and with the return value of find_devices() (as implied by the first parameter: devices)

    Args:
        devices: An instance of tlFactory.EnumerateDevices()
        num_cameras: An integer
        config_files: An integer

    Raises:
        Exception: Any error/exception other than 'no such file or directory'.


    Returns:
        cameras: A dictionary of cameras with ascending lowercase alphabetical letters as keys
    """
    cameras = dict()

    # Create an array of instant cameras for the found devices and avoid exceeding a maximum number of devices.
    instant_camera_array = pylon.InstantCameraArray(min(len(devices), num_cameras))
    cameras["all"] = instant_camera_array
    for i, cam in enumerate(instant_camera_array):
        cam.Attach(tlFactory.CreateDevice(devices[i]))
        print("Camera ", i, "- Using device ", cam.GetDeviceInfo().GetModelName())  # Print camera model number

        cam.Open()
        # 1st camera will be a (ASCII = 97 + 0 = 97), 2nd will be b (ASCII = 97 + 1 = 98) and so on.
        pylon.FeaturePersistence.Load(config_files[chr(97 + i)], cam.GetNodeMap())
        cameras[chr(97 + i)] = cam

    return cameras


def initialize_histograms(bins, num_cameras=2, line_width=3):
    """
    Initializes histogram matplotlib.pyplot figures/subplots

    Args:
        devices: An instance of tlFactory.EnumerateDevices()
        num_cameras: An integer
        config_files: An integer

    Raises:
        Exception: Any error/exception other than 'no such file or directory'.

    Returns:
        histograms: A dictionary of histograms with ascending lowercase alphabetical letters (that match cameras) as keys
    """
    histograms = dict()
    lines = {
        "intensities": dict(),
        "maxima": dict(),
        "averages": dict(),
        "stdevs": dict(),
        "max_vert": dict(),
    }

    fig, (histograms["a"], histograms["b"]) = plt.subplots(1, 2, figsize=(12, 9))  # Initialize plots/subplots
    if num_cameras != 2:
        raise Exception("Unfortunately, this version currently only supports exactly two histograms.")

    for i in range(num_cameras):
        camera_identifier = chr(97 + i)
        histograms[camera_identifier].set_title('Camera ' + camera_identifier.capitalize())
        histograms[camera_identifier].set_xlabel('Bin')
        histograms[camera_identifier].set_ylabel('Frequency')

        lines["intensities"][camera_identifier], = histograms[camera_identifier]\
            .plot(np.arange(bins), np.zeros((bins, 1)), c='k', lw=line_width, label='intensity')

        lines["maxima"][camera_identifier], = histograms[camera_identifier]\
            .plot(np.arange(bins), np.zeros((bins, 1)), c='g', lw=1, label='maximum')

        lines["averages"][camera_identifier], = histograms[camera_identifier]\
            .plot(np.arange(bins), np.zeros((bins, 1)), c='b', linestyle='dashed', lw=1, label='average')

        lines["stdevs"][camera_identifier], = histograms[camera_identifier] \
            .plot(np.arange(bins), np.zeros((bins, 1)), c='r', linestyle='dotted', lw=2, label='stdev')

        lines["max_vert"][camera_identifier] = histograms[camera_identifier]\
            .axvline(0, color='b', linestyle='solid', linewidth=2)

        histograms[camera_identifier].set_xlim(0, bins - 1)
        histograms[camera_identifier].set_ylim(0, 1)
        histograms[camera_identifier].grid(True)
        histograms[camera_identifier].axvline(0, color='b', linestyle='solid', linewidth=2)


    plt.ion()  # Turn the interactive mode on.
    plt.show()

    return fig, histograms, lines

def stream_cam_to_histograms(cams_dict, figure, histograms_dict, lines, bins=4096):
    """
    Description.

    Args:
        example_parameter_a: Description

        example_parameter_b: Description
    Returns:
        abc: An integer
    Raises:
        Exception: Any error/exception other than 'no such file or directory'.
    """

    """Starts grabbing for all cameras starting with index 0. The grabbing is started for one camera after the other.
        That's why the images of all cameras are not taken at the same time. However, a hardware trigger setup can be used
        to cause all cameras to grab images synchronously. According to their default configuration, the cameras are set up 
        for free-running continuous acquisition."""

    cameras = cams_dict["all"]
    cam_a = cams_dict["a"]
    cam_b = cams_dict["b"]

    cameras.StartGrabbing()
    converter = pylon.ImageFormatConverter()

    # Converting to opencv bgr format
    # converter.OutputPixelFormat = pylon.PixelType_BGR12packed
    # converter.OutputPixelFormat = pylon.PixelType_Mono12packed
    # converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputPixelFormat = pylon.PixelType_RGB16packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    while cameras.IsGrabbing():
        # Configure video capture
        capture_a = cv2.VideoCapture(0)
        capture_a.set(cv2.CAP_PROP_FORMAT, cv2.CV_16U)
        capture_a.open(0)

        capture_b = cv2.VideoCapture(1)
        capture_a.set(cv2.CAP_PROP_FORMAT, cv2.CV_16U)
        capture_b.open(1)

        grabResult_a = cam_a.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        grabResult_b = cam_b.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)



        if grabResult_a.GrabSucceeded() and grabResult_b.GrabSucceeded():
            # Access the image data
            image_a = converter.Convert(grabResult_a)
            image_buffer_a = image_a.GetBuffer()
            shape_a = (image_a.GetHeight(), image_a.GetWidth(), 3)
            img_a = np.ndarray(buffer=image_buffer_a, shape=shape_a, dtype=np.uint16)

            # img_a = image_a.GetArray()

            image_b = converter.Convert(grabResult_b)
            image_buffer_b = image_b.GetBuffer()
            shape_b = (image_b.GetHeight(), image_b.GetWidth(), 3)
            img_b = np.ndarray(buffer=image_buffer_b, shape=shape_b, dtype=np.uint16)

            numPixels_a, numPixels_b = np.prod(img_a.shape[:2]), np.prod(img_b.shape[:2])


            gray_img_a, gray_img_b = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)


            # According to opencv's documentation 'OpenCV function is faster than (around 40X) than np.histogram().'
            # Source: https://docs.opencv.org/master/d1/db7/tutorial_py_histogram_begins.html
            histogram_a = cv2.calcHist([gray_img_a], [0], None, [bins], [0, 4095]) / numPixels_a
            histogram_b = cv2.calcHist([gray_img_b], [0], None, [bins], [0, 4095]) / numPixels_b

            lineGray_a = lines["intensities"]["a"]
            lineGray_b = lines["intensities"]["b"]

            lineMaximum_a = lines["maxima"]["a"]
            lineMaximum_b = lines["maxima"]["b"]

            lineAverage_a = lines["averages"]["a"]
            lineAverage_b = lines["averages"]["b"]

            lineStdev_a = lines["stdevs"]["a"]
            lineStdev_b = lines["stdevs"]["b"]

            vertline_x_of_y_max_a = lines["max_vert"]["a"]
            vertline_x_of_y_max_b = lines["max_vert"]["b"]

            lineGray_a.set_ydata(histogram_a)  # Camera A intensity
            lineGray_b.set_ydata(histogram_b)  # Camera B intensity


            maximum_a = np.nanmax(histogram_a)
            maximum_b = np.nanmax(histogram_b)

            indices_a = list(range(0, bins))
            indices_b = list(range(0, bins))

            # Convert histogram to simple list
            hist_a = [val[0] for val in histogram_a]
            hist_b = [val[0] for val in histogram_b]

            # Descending sort-by-key with histogram value as key
            s_a = [(x, y) for y, x in sorted(zip(hist_a, indices_a), reverse=True)]
            s_b = [(x, y) for y, x in sorted(zip(hist_b, indices_b), reverse=True)]

            # Index of highest peak in histogram
            index_of_highest_peak_a = s_a[0][0]
            index_of_highest_peak_b = s_b[0][0]

            average_a = np.average(histogram_a)
            average_b = np.average(histogram_b)

            stdev_a = np.nanstd(histogram_a)
            stdev_b = np.nanstd(histogram_b)

            lineMaximum_a.set_ydata(maximum_a)
            lineAverage_a.set_ydata(average_a)
            lineStdev_a.set_ydata(stdev_a)

            lineMaximum_b.set_ydata(maximum_b)
            lineAverage_b.set_ydata(average_b)
            lineStdev_b.set_ydata(stdev_b)

            vertline_x_of_y_max_a.set_xdata(index_of_highest_peak_a)
            vertline_x_of_y_max_b.set_xdata(index_of_highest_peak_b)

            # Labels must be in the following order: intensity, max, avg, stdev (same order their y values are set)
            # Additionally, if location is not specified, legend will jump around
            histograms_dict["a"].legend(
                labels=(
                    "intensity",
                    "maximum %.2f" % maximum_a,
                    "average %.2f" % average_a,
                    "stdev %.2f" % stdev_a,),
                loc="upper right"
            )

            histograms_dict["b"].legend(
                labels=(
                    "intensity",
                    "maximum %.2f" % maximum_b,
                    "average %.2f" % average_b,
                    "stdev %.2f" % stdev_b),
                loc="upper right"
            )

            histograms_dict["a"].set_ylim(top=maximum_a)
            histograms_dict["b"].set_xlim(right=5000)

            histograms_dict["a"].set_xlim(right=5000)
            histograms_dict["b"].set_ylim(top=maximum_b)

            figure.canvas.draw()

            cv2.namedWindow('Camera A', cv2.WINDOW_NORMAL)
            cv2.imshow('Camera A', img_a)

            cv2.namedWindow('Camera B', cv2.WINDOW_NORMAL)
            cv2.imshow('Camera B', img_b)

            k = cv2.waitKey(1)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Hit q button twice to close break nicely.
                # if 0xFF == ord('q'):  # Hit q button twice to close break nicely.
                cam_a.StopGrabbing()
                cam_b.StopGrabbing()
                break

        grabResult_a.Release()
        grabResult_b.Release()

    return


