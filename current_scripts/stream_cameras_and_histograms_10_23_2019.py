"""How to grab and process images from multiple cameras using the CInstantCameraArray class. The CInstantCameraArray
class represents an array of instant camera objects. It provides almost the same interface as the instant camera for
grabbing. The main purpose of the CInstantCameraArray is to simplify waiting for images and camera events of multiple
cameras in one thread. This is done by providing a single RetrieveResult method for all cameras in the array.
Alternatively, the grabbing can be started using the internal grab loop threads of all cameras in the
CInstantCameraArray. The grabbed images can then be processed by one or more  image event handlers. """

from pypylon import genicam, pylon  # Import relevant pypylon packages/modules
import matplotlib.pyplot as plt  # For plotting live histogram
import os, sys, cv2, argparse  # Need for reading files, exiting system, streaming video, and command line arguments
import numpy as np  # Pixel math, among other array operations
import traceback # exception handling
from numpy.linalg import inv


def foo(param1):
    """
    :param param1: this better be an int
    :return: Boolean true or false
    """

parent_directory = os.path.dirname(os.getcwd())  # String representing parent directory of current working directory
camera_configurations_folder = parent_directory + "\camera_configuration_files"
camera_a_configuration = camera_configurations_folder + "\/23170624_setup_Oct15.pfs"
camera_b_configuration = camera_configurations_folder + "\/23170624_setup_Oct15.pfs"


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file',
    help='Path to video file (if not using camera)')
parser.add_argument('-c', '--color', type=str, default='gray',
    help='Color space: "gray" (default) or "rgb"')
parser.add_argument('-b', '--bins', type=int, default=16,
    help='Number of bins per channel (default 16)')
parser.add_argument('-w', '--width', type=int, default=0,
    help='Resize video to specified width in pixels (maintains aspect)')

args = vars(parser.parse_args()) # Parse user arguments into a dictionary
color, bins,  resizeWidth = args['color'], args['bins'], args['width'] # Pulling values from argument map


"""Limits the amount of cameras used for grabbing. It is important to manage the available bandwidth when grabbing with 
multiple cameras. This applies, for instance, if two GigE cameras are connected to the same network adapter via a 
switch. To manage the bandwidth, the GevSCPD interpacket delay parameter and the GevSCFTD transmission delay parameter 
can be set for each GigE camera device. The "Controlling Packet Transmission Timing with the Interpacket and Frame 
Transmission Delays on Basler GigE Vision Cameras" Application Notes (AW000649xx000) provide more information about 
this topic. The bandwidth used by a FireWire camera device can be limited by adjusting the packet size."""
maxCamerasToUse = 2 # Number of cameras we plan on using
exitCode = 0 # The exit code of the sample application.


count = 0
try:
    tlFactory = pylon.TlFactory.GetInstance() # Get the transport layer factory.
    devices = tlFactory.EnumerateDevices() # Get all attached devices and exit application if no device is found.

    if len(devices) == 0:
        raise RuntimeError("No cameras present.")

    # Create an array of instant cameras for the found devices and avoid exceeding a maximum number of devices.
    cameras = pylon.InstantCameraArray(min(len(devices), maxCamerasToUse))
    cam_a, cam_b = None, None # First we have to declare the camera variables. Then we'll set them in the for loop

    # Create and attach all Pylon Devices.
    for i, cam in enumerate(cameras):
        cam.Attach(tlFactory.CreateDevice(devices[i]))
        print("Camera ", i, "- Using device ", cam.GetDeviceInfo().GetModelName())  # Print camera model number
        cam.Open()

        if i == 0:
            pylon.FeaturePersistence.Load(camera_a_configuration, cam.GetNodeMap())
            cam_a = cam
        if i == 1:
            pylon.FeaturePersistence.Load(camera_b_configuration, cam.GetNodeMap())
            cam_b = cam
        elif i > 1:
            print("You have connected an additional camera, but have not provided a third configuration file.")


    fig, (cam_a_hist, cam_b_hist) = plt.subplots(1, 2, figsize = (12, 9))  # Initialize plots/subplots
    cam_a_hist.set_title('Camera A')
    cam_a_hist.set_xlabel('Bin')
    cam_a_hist.set_ylabel('Frequency')

    cam_b_hist.set_title('Camera B')
    cam_b_hist.set_xlabel('Bin')
    cam_b_hist.set_ylabel('Frequency')

    # Initialize plot line object(s). Turn on interactive plotting and show plot.
    lw = 3 # Line width and

    lineGray_a, = cam_a_hist.plot(np.arange(bins), np.zeros((bins, 1)), c='k', lw=lw, label='intensity')
    lineGray_b, = cam_b_hist.plot(np.arange(bins), np.zeros((bins, 1)), c='k', lw=lw, label='intensity')

    lineMaximum_a, = cam_a_hist.plot(np.arange(bins), np.zeros((bins, 1)), c='g', lw=1, label='maximum')
    lineMaximum_b, = cam_b_hist.plot(np.arange(bins), np.zeros((bins, 1)), c='g', lw=1, label='maximum')

    lineAverage_a, = cam_a_hist.plot(np.arange(bins), np.zeros((bins, 1)), c='b', linestyle='dashed', lw=1,
                                   label='average')
    lineAverage_b, = cam_b_hist.plot(np.arange(bins), np.zeros((bins, 1)), c='b', linestyle='dashed', lw=1,
                                   label='average')
    lineStdev_a, = cam_a_hist.plot(np.arange(bins), np.zeros((bins, 1)), c='r', linestyle='dotted', lw=2, label='stdev')
    lineStdev_b, = cam_b_hist.plot(np.arange(bins), np.zeros((bins, 1)), c='r', linestyle='dotted', lw=2, label='stdev')

    min_xlim, max_xlim = plt.xlim()
    min_ylim, max_ylim = plt.ylim()

    cam_a_hist.set_xlim(0, bins - 1)
    cam_a_hist.set_ylim(0, 1)
    legend_lines_a = cam_a_hist.legend(loc=1)
    cam_a_hist.grid(True)

    cam_b_hist.set_xlim(0, bins - 1)
    cam_b_hist.set_ylim(0, 1)
    legend_lines_b = cam_b_hist.legend(loc=1)
    cam_b_hist.grid(True)

    vertline_x_of_y_max_a = cam_a_hist.axvline(0, color='b', linestyle='solid', linewidth=2)
    vertline_x_of_y_max_b = cam_b_hist.axvline(0, color='b', linestyle='solid', linewidth=2)

    plt.ion()  # Turn the interactive mode on.
    plt.show()

    """Starts grabbing for all cameras starting with index 0. The grabbing is started for one camera after the other.
    That's why the images of all cameras are not taken at the same time. However, a hardware trigger setup can be used
    to cause all cameras to grab images synchronously. According to their default configuration, the cameras are set up 
    for free-running continuous acquisition."""
    cameras.StartGrabbing()
    converter = pylon.ImageFormatConverter()

    # Converting to opencv bgr format
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    while cameras.IsGrabbing():
        # Configure video capture
        capture_a = cv2.VideoCapture(0)
        capture_a.open(0)

        capture_b = cv2.VideoCapture(1)
        capture_b.open(1)

        grabResult_a = cam_a.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        grabResult_b = cam_b.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        print("grabResult_a")
        print(grabResult_a)
        print("grabResult_a.GetPixelType()")
        print(grabResult_a.GetPixelType())
        print("type(grabResult_a.GetPixelType())")
        print(type(grabResult_a.GetPixelType()))
        #print("grabResult_a.GetSizeX()")
        #print(grabResult_a.GetSizeX())
        print()


        print("grabResult_b")
        print(grabResult_b)
        print("grabResult_b.GetPixelType()")
        print(grabResult_b.GetPixelType())
        print("type(grabResult_b.GetPixelType())")
        print(type(grabResult_b.GetPixelType()))
        #print("grabResult_a.GetSizeX()")
        #print(grabResult_a.GetSizeX())
        print()




        if grabResult_a.GrabSucceeded() and grabResult_b.GrabSucceeded():

            # Access the image data
            image_a = converter.Convert(grabResult_a)
            img_a = image_a.GetArray()

            image_b = converter.Convert(grabResult_b)
            img_b = image_b.GetArray()

            numPixels_a, numPixels_b = np.prod(img_a.shape[:2]), np.prod(img_b.shape[:2])

            print("    numPixels_a = %s \n    numPixels_b = %s\n" % (numPixels_a, numPixels_b))

            gray_img_a, gray_img_b = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

            print("img_a.shape)")
            print(img_a.shape)


            print("gray_img_a.shape)")
            print(gray_img_a.shape)

            print("gray_img_a")
            print(gray_img_a)


            """According to opencv's documentation 'OpenCV function is faster than (around 40X) than np.histogram().'
            Source: https://docs.opencv.org/master/d1/db7/tutorial_py_histogram_begins.html"""
            histogram_a = cv2.calcHist([gray_img_a], [0], None, [16], [0, 255]) / numPixels_a
            histogram_b = cv2.calcHist([gray_img_b], [0], None, [16], [0, 255]) / numPixels_b
            """
            cv2.calcHist(images, channels, mask, histSize, ranges, hist=None, accumulate=None)
            images:
                Source arrays. They all should have the same depth, CV_8U or CV_32F , and the same size. Each of 
                them can have an arbitrary number of channels
            channels:
                List of the dims channels used to compute the histogram. The first array channels are numerated 
                from 0 to images[0].channels()-1 , the second array channels are counted from images[0].channels() to 
                images[0].channels() + images[1].channels()-1, and so on.
            mask:
                Optional mask. If the matrix is not empty, it must be an 8-bit array of the same size as images[i] . 
                The non-zero mask elements mark the array elements counted in the histogram
            hist:
                Output histogram, which is a dense or sparse dims -dimensional array
            dims: 
                Histogram dimensionality that must be positive and not greater than CV_MAX_DIMS (equal to 32 in the 
                current OpenCV version).
            """

            print("histogram_a")
            print(histogram_a)

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
            cam_a_hist.legend(
                labels = (
                    "intensity",
                    "maximum %.2f" % maximum_a,
                    "average %.2f" % average_a,
                    "stdev %.2f" % stdev_a,),
                loc = "upper right"
            )

            cam_b_hist.legend(
                labels = (
                    "intensity",
                    "maximum %.2f" % maximum_b,
                    "average %.2f" % average_b,
                    "stdev %.2f" % stdev_b),
                loc = "upper right"
            )

            fig.canvas.draw()

            cv2.namedWindow('Camera A', cv2.WINDOW_NORMAL)
            cv2.imshow('Camera A', img_a)

            cv2.namedWindow('Camera B', cv2.WINDOW_NORMAL)
            cv2.imshow('Camera B', img_b)

            k = cv2.waitKey(1)
            if cv2.waitKey(1) & 0xFF == ord('q'): # Hit q button twice to close break nicely.
            #if 0xFF == ord('q'):  # Hit q button twice to close break nicely.
                cam_a.StopGrabbing()
                cam_b.StopGrabbing()
                break
        grabResult_a.Release()
        grabResult_b.Release()

except Exception as e:
    # Error handling
    print("An exception occurred:")
    traceback.print_exc()
    exitCode = 1

# Comment the following two lines to disable waiting on exit.
sys.exit(exitCode)
