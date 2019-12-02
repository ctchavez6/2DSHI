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
import png
from PIL import Image
from io import BytesIO
from datetime import datetime

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
        num_cameras (int): An integer
        config_files: An integer
    Raises:
        Exception: Any error/exception other than 'no such file or directory'.
    Returns:
        dict: A dictionary of cameras with ascending lowercase alphabetical letters as keys.
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


def initialize_histograms(bins, num_cameras=2, line_width=3):
    """
    Initializes histogram matplotlib.pyplot figures/subplots.

    Args:
        bins: An instance of tlFactory.EnumerateDevices()
        num_cameras: An integer
        line_width: An integer
    Raises:
        Exception: Any error/exception other than 'no such file or directory'.
    Returns:
        dict: A dictionary of histograms with ascending lowercase alphabetical letters (that match cameras) as keys
    """
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


    if num_cameras != 2:
        raise Exception("Unfortunately, this version currently only supports exactly two histograms.")

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
            .axvline(-100, color='b', linestyle='solid', linewidth=1)

        lines["avg+sigma"][camera_identifier] = stream_subplots[camera_identifier] \
            .axvspan(-100, -100, alpha=0.5, color='#f5beba')

        lines["avg-sigma"][camera_identifier] = stream_subplots[camera_identifier] \
            .axvspan(-100, -100, alpha=0.5, color='#f5beba')

        stream_subplots[camera_identifier].set_xlim(-100, bins - 1 + 100)
        stream_subplots[camera_identifier].grid(True)
        stream_subplots[camera_identifier].set_autoscale_on(False)
        stream_subplots[camera_identifier].set_ylim(bottom=0, top=1)

    plt.ion()  # Turn the interactive mode on.
    figs = dict()
    figs["a"], figs["b"] = fig_a, fig_b

    return figs, stream_subplots, lines


def update_histogram(histogram_dict, lines_dict, identifier, bins, raw_2d_array,threshold=1.2):
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


def convert_to_16_bit(image_array, original_bit_depth=12):
    """
    Takes an image array and represents it as 16 bit by multiplying all the values by the corresponding integer and
    specifying the bit depth in the creation of a new 2D Numpy Array.

    Args:
        image_array (numpy.ndarray): The original image array.
    Returns:
        numpy.ndarray: The same image represented
    """
    if original_bit_depth < 16:
        return np.array(image_array * 2**(16-original_bit_depth), dtype=np.uint16).astype(np.uint16)
    else:
        raise Exception('Original Bit Depth was greater than or equal to 16')

def convert_to_8_bit(image_array, original_bit_depth=12, grayscale_to_rgb=False):
    """
    TODO Add documentation.
    """
    if original_bit_depth == 12:
        return np.array(image_array/16, dtype=np.uint8)
    return None


def resize_img(image_array, new_width, new_height):
    """
    TODO Add documentation.
    """
    return cv2.resize(image_array, (new_width, new_height), interpolation=cv2.INTER_AREA)


def save_img(filename, directory, image, sixteen_bit=True):
    """
    TODO Add documentation.
    """
    os.chdir(directory)
    if sixteen_bit:
        image = Image.fromarray(image)
        image.save(filename, compress_level=0)
    else:
        cv2.imwrite(filename, image.astype(np.uint16))
    os.chdir(directory)


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

    img_a_8bit_gray = convert_to_8_bit(raw_array_a)
    img_b_8bit_gray = convert_to_8_bit(raw_array_b)

    img_a_8bit_resized = cv2.cvtColor((resize_img(img_a_8bit_gray, hist_width, hist_height)), cv2.COLOR_GRAY2BGR)
    img_b_8bit_resized = cv2.cvtColor((resize_img(img_b_8bit_gray, hist_width, hist_height)), cv2.COLOR_GRAY2BGR)

    return np.vstack((np.hstack((hist_img_a, img_a_8bit_resized)), np.hstack((hist_img_b, img_b_8bit_resized))))

def set_plot_upper_bound(minimum_upper_bound, upper_bound_factor, maximum, plot):
    """
    Adjusts the y-limit/upper-bound of a matplotlib.pyplot.subplot object.

    Args:
        minimum_upper_bound: In the absence of data, default upper bound.
        upper_bound_factor: Factor to multiply the data maximum by.
        maximum: Image intensity histogram maximum.
        plot (matplotlib.pyplot.subplot): Subplot that represents camera histogram.
    """
    if maximum > minimum_upper_bound:
        plot.set_ylim(bottom=0.000000, top=maximum * upper_bound_factor)
    else:
        plot.set_ylim(bottom=0.000000, top=minimum_upper_bound)


def reset_images_directory(saved_imgs_directory):
    """
    Deletes any .tiff or .png image files created during previous runs that were saved in the specified directory.

    Args:
        saved_imgs_directory: Path to the directory where you'd like to clear out .tiff/.png files.
    """
    if os.path.exists(saved_imgs_directory):
        filelist = [f for f in os.listdir(saved_imgs_directory) if (f.endswith(".tiff") or f.endswith(".png"))]
        for f in filelist:
            os.remove(os.path.join(saved_imgs_directory, f))
    else:
        try:
            os.mkdir(saved_imgs_directory)
        except OSError:
            print("Creation of the directory %s failed" % saved_imgs_directory)

def clear_videos(video_directory):
    """
    Deletes any .tiff or .png image files created during previous runs that were saved in the specified directory.
    """
    if os.path.exists(video_directory):
        filelist = [f for f in os.listdir(video_directory) if (f.endswith(".avi") or f.endswith(".mp4"))]
        for f in filelist:
            os.remove(os.path.join(video_directory, f))
    else:
        try:
            os.mkdir(video_directory)
        except OSError:
            print("Creation of the directory %s failed" % video_directory)


def create_and_save_videos(cam_a_frames_direc, cam_b_frames_direc, videos_directory, cams_by_hists_direc):
    """
    By iterating through all the saved frames, constructs a video for each camera.
    Args:
        cam_a_frames_direc: Directory that holds all the saved images for Camera A
        cam_b_frames_direc: Directory that holds all the saved images for Camera B
        videos_directory: Directory where all the video files will be saved
    """
    initial_directory = os.getcwd()

    cam_a_saved_img_files = [file for file in os.listdir(cam_a_frames_direc)]
    cam_b_saved_img_files = [file for file in os.listdir(cam_b_frames_direc)]
    cam_by_his_img_files = [file for file in os.listdir(cams_by_hists_direc)]

    height_a, width_a, layers_a = cv2.imread(os.path.join(cam_a_frames_direc, cam_a_saved_img_files[0])).shape
    height_b, width_b, layers_b = cv2.imread(os.path.join(cam_b_frames_direc, cam_b_saved_img_files[0])).shape
    height_h, width_h, layers_h = cv2.imread(os.path.join(cams_by_hists_direc, cam_by_his_img_files[0])).shape
    # The second parameter, where I use 0, seems to be the default codec. Works well.
    video_a = cv2.VideoWriter(os.path.join(videos_directory, 'video_a.avi'), 0, 10, (height_a, width_a))
    video_b = cv2.VideoWriter(os.path.join(videos_directory, 'video_b.avi'), 0, 10, (height_b, width_b))
    cams_hist_writer = cv2.VideoWriter(
        filename=os.path.join(videos_directory,'cams_with_histogram_representations.avi'),
        fourcc=cv2.VideoWriter_fourcc(*'DIVX'),
        fps=10,
        frameSize=(1000, 1000),
        isColor=1)

    os.chdir(videos_directory)

    for frame_a, frame_b, hist_frame in zip(cam_a_saved_img_files, cam_b_saved_img_files, cam_by_his_img_files):
        video_a.write(cv2.imread(os.path.join(cam_a_frames_direc, frame_a)))
        video_b.write(cv2.imread(os.path.join(cam_b_frames_direc, frame_b)))
        cams_hist_writer.write(cv2.imread(os.path.join(cams_by_hists_direc, hist_frame)))

    cv2.destroyAllWindows()  # Do I need this?
    video_a.release()
    video_b.release()
    cams_hist_writer.release()
    os.chdir(initial_directory)


def clear_prev_run():
    """
    Creates a new directory that corresponds to today's date and time to save the videos/images in. Unlikely, but if
    for some reason two runs happen within the same minute, overwrites first run to keep latest.

    Returns:
        TODO Convert to dictionary? Separate Functionality?
    """
    now = datetime.now()
    current_datetime = now.strftime("%Y_%m_%d__%H_%M")

    data_directory = os.path.join("D:\\2DSHI_Runs", current_datetime)
    if not os.path.exists(data_directory):
        os.mkdir(data_directory)
    camera_a_frames_directory = data_directory + "\\cam_a_frames"
    camera_b_frames_directory = data_directory + "\\cam_b_frames"
    cams_by_hists_directory = data_directory + "\\histogram_streams"
    videos_directory = data_directory + "\\videos"

    reset_images_directory(camera_a_frames_directory)
    reset_images_directory(camera_b_frames_directory)
    reset_images_directory(cams_by_hists_directory)
    clear_videos(videos_directory)
    return camera_a_frames_directory, camera_b_frames_directory, cams_by_hists_directory, videos_directory


def stream_cam_to_histograms(cams_dict, figures, histograms_dict, lines, bins=4096, frame_break=1000, save_imgs=False, save_vids=False):
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
        frame_break: Limit of frames to record. Default 10.
        fps: Frames per second saved video will play at. Default 1.
    """
    frame_count = 0
    cams_dict["all"].StartGrabbing()

    cam_w_histogram_frames, camera_a_frames_as_16bit, camera_b_frames_as_16bit = [], [], []
    raw_cam_a_frames, raw_cam_b_frames = [], []

    while cams_dict["all"].IsGrabbing() and frame_count != frame_break and not (cv2.waitKey(1) & 0xFF == ord('q')):
        grab_result_a = cams_dict["a"].RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        grab_result_b = cams_dict["b"].RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if grab_result_a.GrabSucceeded() and grab_result_b.GrabSucceeded():
            frame_count += 1
            raw_image_a = grab_result_a.GetArray()
            raw_image_b = grab_result_b.GetArray()
            raw_cam_a_frames.append(grab_result_a.GetArray())
            raw_cam_b_frames.append(grab_result_b.GetArray())
            #cv2.imshow("cam_a", convert_to_16_bit(raw_image_a, original_bit_depth=12))
            #cv2.imshow("cam_b", convert_to_16_bit(raw_image_b, original_bit_depth=12))
            #cv2.imshow("cam_a", grab_result_a.GetArray()*16)
            #cv2.imshow("cam_b", grab_result_b.GetArray()*16)

            update_histogram(histograms_dict, lines, "a", bins, raw_image_a)
            update_histogram(histograms_dict, lines, "b", bins, raw_image_b)
            figures["a"].canvas.draw()  # Draw updates subplots in interactive mode
            figures["b"].canvas.draw()  # Draw updates subplots in interactive mode
            cam_w_histogram_frames.append(add_histogram_representations(figures["a"], figures["b"], raw_image_a, raw_image_b))
            cv2.imshow("Camera & Histogram Streams", cam_w_histogram_frames[len(cam_w_histogram_frames)-1])

    cams_dict["a"].StopGrabbing()
    cams_dict["b"].StopGrabbing()

    if save_imgs or save_vids #
        camera_a_frames_directory, camera_b_frames_directory, cams_by_hists_direc, videos_directory = clear_prev_run()

        if save_imgs:
            for a, b in zip(raw_cam_a_frames, raw_cam_b_frames):
                camera_a_frames_as_16bit.append(convert_to_16_bit(a, original_bit_depth=12))
                camera_b_frames_as_16bit.append(convert_to_16_bit(b, original_bit_depth=12))

            frame_count = 0

            for frame_a, frame_b, cam_hist in zip(camera_a_frames_as_16bit, camera_b_frames_as_16bit,cam_w_histogram_frames):
                frame_count += 1
                save_img("cam_a_frame_%s.png" % frame_count, camera_a_frames_directory, frame_a)
                save_img("cam_b_frame_%s.png" % frame_count, camera_b_frames_directory, frame_b)
                save_img("cameras_and_histograms_frame_%s.png" % frame_count, cams_by_hists_direc, cam_hist)

        if save_vids:
            # cams all stop grabbing , check later
            create_and_save_videos(camera_a_frames_directory, camera_b_frames_directory, videos_directory, cams_by_hists_direc)
