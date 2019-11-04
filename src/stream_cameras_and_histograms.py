"""


"""
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
import os
# import sys
import cv2
import numpy as np  # Pixel math, among other array operations
import traceback # exception handling
# from numpy.linalg import inv

def grab_frame(cap):
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return frame

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
    stream_subplots = dict()
    lines = {
        "intensities": dict(),
        "maxima": dict(),
        "averages": dict(),
        "stdevs": dict(),
        "max_vert": dict(),
    }
    #fig, (stream_subplots["a"], stream_subplots["b"]) = plt.subplots(1, 2, figsize=(12, 9))  # Initialize plots/subplots
    fig_a = plt.figure(figsize=(5, 5))
    stream_subplots["a"] = fig_a.add_subplot()
    fig_b = plt.figure(figsize=(5, 5))
    stream_subplots["b"] = fig_b.add_subplot()

    """
    stream_subplots["cam_a"] = axs[0, 0]
    stream_subplots["cam_a"].set_title('Camera A')
    stream_subplots["cam_a"].tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    stream_subplots["cam_a"].tick_params(
        axis='y',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off




    stream_subplots["cam_b"] = axs[1, 0]
    stream_subplots["cam_b"].set_title('Camera B')
    stream_subplots["cam_b"].tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    stream_subplots["cam_b"].tick_params(
        axis='y',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    
    """


    if num_cameras != 2:
        raise Exception("Unfortunately, this version currently only supports exactly two histograms.")

    for i in range(num_cameras):
        camera_identifier = chr(97 + i)
        stream_subplots[camera_identifier].set_title('Camera ' + camera_identifier.capitalize())
        stream_subplots[camera_identifier].set_xlabel('Bin')
        stream_subplots[camera_identifier].set_ylabel('Frequency')

        lines["intensities"][camera_identifier], = stream_subplots[camera_identifier]\
            .plot(np.arange(bins), np.zeros((bins, 1)), c='k', lw=line_width, label='intensity')

        lines["maxima"][camera_identifier], = stream_subplots[camera_identifier]\
            .plot(np.arange(bins), np.zeros((bins, 1)), c='g', lw=1, label='maximum')

        lines["averages"][camera_identifier], = stream_subplots[camera_identifier]\
            .plot(np.arange(bins), np.zeros((bins, 1)), c='b', linestyle='dashed', lw=1, label='average')

        lines["stdevs"][camera_identifier], = stream_subplots[camera_identifier] \
            .plot(np.arange(bins), np.zeros((bins, 1)), c='r', linestyle='dotted', lw=2, label='stdev')

        lines["max_vert"][camera_identifier] = stream_subplots[camera_identifier]\
            .axvline(0, color='b', linestyle='solid', linewidth=2)

        stream_subplots[camera_identifier].set_xlim(0, bins - 1)
        stream_subplots[camera_identifier].set_ylim(0, 1)
        stream_subplots[camera_identifier].grid(True)
        stream_subplots[camera_identifier].axvline(0, color='b', linestyle='solid', linewidth=2)
        stream_subplots[camera_identifier].set_autoscale_on(False)

    plt.ion()  # Turn the interactive mode on.
    figs = dict()
    figs["a"], figs["b"] = fig_a, fig_b
    return figs, stream_subplots, lines

def stream_cam_to_histograms(cams_dict, figures, histograms_dict, lines, bins=4096):
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
    converter.OutputPixelFormat = pylon.PixelType_RGB16packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    histograms_dict["a"].set_autoscale_on(False)
    histograms_dict["b"].set_autoscale_on(False)
    histograms_dict["a"].set_ylim(bottom=0, top=1)
    histograms_dict["b"].set_ylim(bottom=0, top=1)



    initial_current_working_directory = os.getcwd()
    camera_b_frames_directory = initial_current_working_directory + "/cam_b_frames"

    if os.path.exists(camera_b_frames_directory):
        filelist = [f for f in os.listdir(camera_b_frames_directory) if f.endswith(".tiff")]
        for f in filelist:
            os.remove(os.path.join(camera_b_frames_directory, f))
    else:
        try:
            os.mkdir(camera_b_frames_directory)
        except OSError:
            print("Creation of the directory %s failed" % camera_b_frames_directory)

    if os.path.exists('camera_a.avi'):
        os.remove('camera_a.avi')
    if os.path.exists('camera_b.avi'):
        os.remove('camera_b.avi')
    if os.path.exists('four_by_four.avi'):
        os.remove('four_by_four.avi')




    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    fourcc_iyuv = cv2.VideoWriter_fourcc('i', 'Y', 'U', 'V')
    fourcc_hists = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    fourcc_ffv1 = cv2.VideoWriter_fourcc('F', 'F', 'V', '1')

    #camera_a_stream_video = cv2.VideoWriter('camera_a.avi', fourcc_ffv1, 1,
    #                           (1200, 1200), True)
    #camera_b_stream_video = cv2.VideoWriter('camera_b.avi', fourcc_ffv1, 1,
    #                                        (1200, 1200), True)

    cams_hist_writer = cv2.VideoWriter('four_by_four.avi', fourcc_hists, 5,
                               (1000, 1000), True)


    frame_count = 0
    cameras_histogram_4x4_frames = []
    camera_a_frames = []
    camera_b_frames = []


    while cameras.IsGrabbing():
        # Configure video capture
        capture_a = cv2.VideoCapture(0)
        capture_a.set(cv2.CAP_PROP_FORMAT, cv2.CV_16U)
        capture_a.open(0)

        capture_b = cv2.VideoCapture(1)
        capture_a.set(cv2.CAP_PROP_FORMAT, cv2.CV_16UC1)

        capture_b.open(1)

        grabResult_a = cam_a.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        grabResult_b = cam_b.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grabResult_a.GrabSucceeded() and grabResult_b.GrabSucceeded():
            frame_count += 1
            print("Frame %s " % frame_count)

            # Access the image data
            image_a = converter.Convert(grabResult_a)
            image_buffer_a = image_a.GetBuffer()
            shape_a = (image_a.GetHeight(), image_a.GetWidth(), 3)
            img_a = np.ndarray(buffer=image_buffer_a, shape=shape_a, dtype=np.uint16)

            frame = cv2.cvtColor(img_a, cv2.COLOR_RGB2GRAY)
            #cv2.imshow("frame", frame)

            camera_a_frames.append(img_a)
            img_a_8bit_500px = (cv2.resize(img_a, (500, 500), interpolation=cv2.INTER_AREA)/ 256).astype('uint8')

            image_b = converter.Convert(grabResult_b)
            image_buffer_b = image_b.GetBuffer()
            shape_b = (image_b.GetHeight(), image_b.GetWidth(), 3)
            img_b = np.ndarray(buffer=image_buffer_b, shape=shape_b, dtype=np.uint16)





            img_b_8bit_500px = (cv2.resize(img_b, (500, 500), interpolation=cv2.INTER_AREA)/ 256).astype('uint8')
            os.chdir(camera_b_frames_directory)
            cv2.imwrite("cam_b_frame_%s.tiff" % frame_count, img_b)
            os.chdir(initial_current_working_directory)
            camera_b_frames.append(img_b)

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

            maximum_a = np.amax(histogram_a)
            maximum_b = np.amax(histogram_b)

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
                    "maximum %.4f" % maximum_a,
                    "average %.4f" % average_a,
                    "stdev %.4f" % stdev_a,),
                loc="upper right"
            )

            histograms_dict["b"].legend(
                labels=(
                    "intensity",
                    "maximum %.4f" % maximum_b,
                    "average %.4f" % average_b,
                    "stdev %.4f" % stdev_b),
                loc="upper right"
            )

            histograms_dict["a"].set_xlim(left=0, right=4096)
            histograms_dict["b"].set_xlim(left=0, right=4096)

            if maximum_a > 0.001:
                histograms_dict["a"].set_ylim(bottom=0.000000, top= maximum_a*1.2)
            else:
                histograms_dict["a"].set_ylim(bottom=0.000000, top= 0.001)


            if maximum_b >  0.001:
                histograms_dict["b"].set_ylim(bottom=0.000000, top=maximum_b*1.2)
            else:
                histograms_dict["b"].set_ylim(bottom=0.000000, top= 0.001)

            figures["a"].canvas.draw()
            figures["b"].canvas.draw()

            hist_img_a = np.fromstring(figures["a"].canvas.tostring_rgb(), dtype=np.uint8, sep='') # convert canvas to image
            hist_img_a = hist_img_a.reshape(figures["a"].canvas.get_width_height()[::-1] + (3,))
            hist_img_a = cv2.cvtColor(hist_img_a, cv2.COLOR_RGB2BGR) # img is rgb, convert to opencv's default bgr

            hist_img_b = np.fromstring(figures["b"].canvas.tostring_rgb(), dtype=np.uint8, sep='') # convert canvas to image
            hist_img_b = hist_img_b.reshape(figures["b"].canvas.get_width_height()[::-1] + (3,))
            hist_img_b = cv2.cvtColor(hist_img_b, cv2.COLOR_RGB2BGR) # img is rgb, convert to opencv's default bgr

            cameras_histograms_4x4 = np.vstack(
                (np.hstack((hist_img_a, img_a_8bit_500px)),
                 np.hstack((hist_img_b, img_b_8bit_500px))))
            cv2.imshow("Camera & Histogram Streams", cameras_histograms_4x4)
            cameras_histogram_4x4_frames.append(cameras_histograms_4x4)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Hit q button twice to close break nicely.
                # if 0xFF == ord('q'):  # Hit q button twice to close break nicely.
                cam_a.StopGrabbing()
                cam_b.StopGrabbing()
                break
            if frame_count == 15:
                "Hit Frame 25, Breaking"
                break
        grabResult_a.Release()
        grabResult_b.Release()

    #for frame in camera_a_frames:
    #    camera_a_stream_video.write(frame)
    #for frame in camera_b_frames:
    #    camera_b_stream_video.write(frame)
    #for frame in cameras_histogram_4x4_frames:
    #    cams_hist_writer.write(frame)


    cams_hist_writer.release()
    #camera_a_stream_video.release()
    #camera_b_stream_video.release()

    video_name = 'video_b.avi'

    images = [img for img in os.listdir(camera_b_frames_directory) if img.endswith(".tiff")]
    frame = cv2.imread(os.path.join(camera_b_frames_directory, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 1, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(camera_b_frames_directory, image)))

    cv2.destroyAllWindows()
    video.release()

    return


