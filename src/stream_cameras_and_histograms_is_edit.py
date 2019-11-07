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
import traceback  # exception handling





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
    fig_a = plt.figure(figsize=(5, 5))
    stream_subplots["a"] = fig_a.add_subplot()
    fig_b = plt.figure(figsize=(5, 5))
    stream_subplots["b"] = fig_b.add_subplot()


    if num_cameras != 2:
        raise Exception("Unfortunately, this version currently only supports exactly two histograms.")

    for i in range(num_cameras):
        camera_identifier = chr(97 + i)
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

        lines["max_vert"][camera_identifier] = stream_subplots[camera_identifier] \
            .axvline(0, color='b', linestyle='solid', linewidth=2)

        stream_subplots[camera_identifier].set_xlim(0, bins - 1)
        stream_subplots[camera_identifier].grid(True)
        stream_subplots[camera_identifier].axvline(0, color='b', linestyle='solid', linewidth=2)
        stream_subplots[camera_identifier].set_autoscale_on(False)
        stream_subplots[camera_identifier].set_ylim(bottom=0, top=1)

    plt.ion()  # Turn the interactive mode on.
    figs = dict()
    figs["a"], figs["b"] = fig_a, fig_b

    return figs, stream_subplots, lines


def update_histogram(histogram_dict, lines_dict, histogram_identifier, calculated_hist, bins, threshold=1.2):
    maximum = np.amax(calculated_hist)
    average = np.average(calculated_hist)
    stdev = np.nanstd(calculated_hist)

    update_histogram_lines(lines_dict, histogram_identifier, calculated_hist, bins)

    histogram_dict[histogram_identifier].legend(
        labels=(
            "intensity",
            "maximum %.4f" % maximum,
            "average %.4f" % average,
            "stdev %.4f" % stdev,),
        loc="upper right"
    )

    if maximum > 0.001:
        histogram_dict[histogram_identifier].set_ylim(bottom=0.000000, top=maximum * threshold)
    else:
        histogram_dict[histogram_identifier].set_ylim(bottom=0.000000, top=0.001)

    histogram_dict[histogram_identifier].canvas.draw()


def update_histogram_lines(lines_dict, identifier, calculated_hist, bins):
    lines_dict["intensities"][identifier].set_ydata(calculated_hist)  # Intensities/Percent of Saturation
    maximum = np.amax(calculated_hist)
    average = np.average(calculated_hist)
    stdev = np.nanstd(calculated_hist)

    indices = list(range(0, bins))
    hist = [val[0] for val in calculated_hist]
    s = [(x, y) for y, x in sorted(zip(hist, indices), reverse=True)]
    index_of_highest_peak = s[0][0]

    lines_dict["maxima"][identifier].set_ydata(maximum)  # Maximums
    lines_dict["averages"][identifier].set_ydata(average)  # Averages
    lines_dict["stdevs"][identifier].set_ydata(stdev)  # Standard Deviations
    lines_dict["max_vert"][identifier].set_xdata(index_of_highest_peak)  # Maximum Indicator as vertical line

def initialize_dual_video_capture(first_channel=0, second_channel=1):
    capture_a, capture_b = cv2.VideoCapture(first_channel), cv2.VideoCapture(second_channel)
    capture_a.set(cv2.CAP_PROP_FORMAT, cv2.CV_16U)
    capture_b.set(cv2.CAP_PROP_FORMAT, cv2.CV_16UC1)
    capture_a.open(first_channel)
    capture_b.open(second_channel)
    return capture_a, capture_b


def save_img(filename, directory, image):
    os.chdir(directory)
    img = np.ndarray(buffer=image.GetBuffer(),
                       shape=(image.GetHeight(), image.GetWidth(), 3),
                       dtype=np.uint16)
    cv2.imwrite(filename, img)
    os.chdir(directory)
    return img

def create_camera_histogram_4x4(figure_a, figure_b, cam_img_a, cam_img_b):
    hist_img_a = np.fromstring(figure_a.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    hist_img_b = np.fromstring(figure_b.canvas.tostring_rgb(), dtype=np.uint8, sep='')  # convert  to image

    hist_img_a = hist_img_a.reshape(figure_a.canvas.get_width_height()[::-1] + (3,))
    hist_img_b = hist_img_b.reshape(figure_b.canvas.get_width_height()[::-1] + (3,))

    hist_img_a = cv2.cvtColor(hist_img_a, cv2.COLOR_RGB2BGR)  # img is rgb, convert to opencv's default bgr
    hist_img_b = cv2.cvtColor(hist_img_b, cv2.COLOR_RGB2BGR)  # img is rgb, convert to opencv's default bgr

    img_a_8bit_500px = (cv2.resize(cam_img_a, (500, 500), interpolation=cv2.INTER_AREA) / 256).astype('uint8')
    img_b_8bit_500px = (cv2.resize(cam_img_b, (500, 500), interpolation=cv2.INTER_AREA) / 256).astype('uint8')

    return np.vstack(
        (np.hstack((hist_img_a, img_a_8bit_500px)),
         np.hstack((hist_img_b, img_b_8bit_500px))))

def set_plot_upper_bound(minimum_upper_bound, upper_bound_factor, maximum, plot):
    if maximum > minimum_upper_bound:
        plot.set_ylim(bottom=0.000000, top=maximum * upper_bound_factor)
    else:
        plot.set_ylim(bottom=0.000000, top=minimum_upper_bound)


def reset_images_directory(saved_imgs_directory, extension=".tiff"):
    """
    Deletes any image files created during previous runs, and creates a directory for


    """
    if os.path.exists(saved_imgs_directory):
        filelist = [f for f in os.listdir(saved_imgs_directory) if f.endswith(extension)]
        for f in filelist:
            os.remove(os.path.join(saved_imgs_directory, f))
    else:
        try:
            os.mkdir(saved_imgs_directory)
        except OSError:
            print("Creation of the directory %s failed" % saved_imgs_directory)

def clear_videos(list_of_videos, video_directory):
    """
    Deletes any video files created during previous runs.


    """
    initial_directory = os.getcwd()
    if os.path.exists(video_directory):
        os.chdir(video_directory)
        for video in list_of_videos:
            if os.path.exists(video):

                os.remove(video)
    else:
        try:
            os.mkdir(video_directory)
        except OSError:
            print("Creation of the directory %s failed" % video_directory)
    os.chdir(initial_directory)


def create_and_save_videos(cam_a_frames_direc, cam_b_frames_direc, videos_directory):
    """
    By iterating through all the saved frames, constructs a video for each camera.

    Args:
        cam_a_frames_direc: Directory that holds all the saved images for Camera A
        cam_b_frames_direc: Directory that holds all the saved images for Camera B
        videos_directory: Directory where all the video files will be saved
    """
    initial_directory = os.getcwd()

    cam_a_saved_img_files = [img for img in os.listdir(cam_a_frames_direc) if img.endswith(".tiff")]
    cam_b_saved_img_files = [img for img in os.listdir(cam_b_frames_direc) if img.endswith(".tiff")]

    height_a, width_a, layers_a = cv2.imread(os.path.join(cam_a_frames_direc, cam_a_saved_img_files[0])).shape
    height_b, width_b, layers_b = cv2.imread(os.path.join(cam_b_frames_direc, cam_b_saved_img_files[0])).shape

    video_a = cv2.VideoWriter('video_a.avi', 0, 1, (height_a, width_a))
    video_b = cv2.VideoWriter('video_b.avi', 0, 1, (height_b, width_b))

    os.chdir(videos_directory)

    for frame_a, frame_b in zip(cam_a_saved_img_files, cam_b_saved_img_files):
        video_a.write(cv2.imread(os.path.join(cam_a_frames_direc, frame_a)))
        video_b.write(cv2.imread(os.path.join(cam_b_frames_direc, frame_b)))

    cv2.destroyAllWindows()
    video_b.release()
    os.chdir(initial_directory)

def get_pylon_image_converter():
    """
    Creates and returns an instance of a pylon.ImageFormatConverter() after updating the Output Pixel Format as well
    as the Output Bit Alignment.

    Returns:
        converter: pylon.ImageFormatConverter() object with RGB16packed Pixel Format and MsbAligned Output Bit Alignment
    """
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_RGB16packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    return converter

def clear_previous_run():
    initial_current_working_directory = os.getcwd()
    possibly_pre_existing_videos = ['camera_a.avi', 'camera_b.avi', 'four_by_four.avi']
    camera_a_frames_directory = initial_current_working_directory + "/cam_a_frames"
    camera_b_frames_directory = initial_current_working_directory + "/cam_b_frames"
    videos_directory = initial_current_working_directory + "/videos"
    reset_images_directory(camera_a_frames_directory)
    reset_images_directory(camera_b_frames_directory)
    clear_videos(possibly_pre_existing_videos, videos_directory)
    return camera_a_frames_directory, camera_b_frames_directory, videos_directory

def stream_cam_to_histograms(cams_dict, figures, histograms_dict, lines, bins=4096, frame_break=100, fps=1):
    """
    Starts grabbing for all cameras starting with index 0. The grabbing is started for one camera after the other.
    That's why the images of all cameras are not taken at the same time. However, a hardware trigger setup can be used
    to cause all cameras to grab images synchronously. According to their default configuration, the cameras are set up
    for free-running continuous acquisition. Also saves all the images and videos.

    Args:
        cams_dict: Description
        figures: Description
        histograms_dict: Description
        lines: Description
        bins: Description
        frame_break: Limit of frames to record. Default 100.
        fps: Frames per second saved video will play at. Default 1.
    """
    frame_count = 0
    cams_dict["all"].StartGrabbing()
    converter = get_pylon_image_converter()

    camera_a_frames_directory, camera_b_frames_directory, videos_directory = clear_previous_run()
    cameras_histogram_4x4_frames, camera_a_frames, camera_b_frames = [], [], []

    cams_hist_writer = cv2.VideoWriter('four_by_four.avi',  # File Name
                                       cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),  # Codec
                                       fps,  # Frames Per Second
                                       (1000, 1000),  # Dimensions
                                       True)  # Start

    while cams_dict["all"].IsGrabbing() and frame_count != frame_break and not (cv2.waitKey(1) & 0xFF == ord('q')):
        initialize_dual_video_capture()

        grab_result_a = cams_dict["a"].RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        grab_result_b = cams_dict["b"].RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grab_result_a.GrabSucceeded() and grab_result_b.GrabSucceeded():
            frame_count += 1
            print("Frame %s " % frame_count)

            image_a, image_b = converter.Convert(grab_result_a), converter.Convert(grab_result_b)

            img_a = save_img("cam_a_frame_%s.tiff" % frame_count, camera_a_frames_directory, image_a)
            img_b = save_img("cam_b_frame_%s.tiff" % frame_count, camera_b_frames_directory, image_b)

            gray_img_a, gray_img_b = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

            histogram_a = cv2.calcHist([gray_img_a], [0], None, [bins], [0, 4095]) / np.prod(img_a.shape[:2])
            histogram_b = cv2.calcHist([gray_img_b], [0], None, [bins], [0, 4095]) / np.prod(img_b.shape[:2])

            update_histogram(histograms_dict, lines, "a", histogram_a, bins)
            update_histogram(histograms_dict, lines, "b", histogram_b, bins)

            cameras_histograms_4x4 = create_camera_histogram_4x4(figures["a"], figures["a"], img_a, img_b)
            cv2.imshow("Camera & Histogram Streams", cameras_histograms_4x4)

            camera_a_frames.append(img_a)
            camera_b_frames.append(img_b)
            cameras_histogram_4x4_frames.append(cameras_histograms_4x4)

    cams_dict["a"].StopGrabbing()
    cams_dict["b"].StopGrabbing()
    create_and_save_videos(camera_a_frames_directory, camera_b_frames_directory, videos_directory)
    cams_hist_writer.release()



