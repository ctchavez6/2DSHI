from pypylon import genicam, pylon  # Import relevant pypylon packages/modules
import cv2
from image_processing import bit_depth_conversion as bdc
from image_processing import stack_images as stack
from histocam import histocam
from coregistration import img_characterization as ic
from coregistration import find_gaussian_profile as fgp
import numpy as np
import matplotlib.pyplot as plt
import sys
from image_processing import img_algebra as ia

EPSILON = sys.float_info.epsilon  # Smallest possible difference
sixteen_bit_max = (2 ** 16) - 1

#plt.ion()  # Turn the interactive mode on.


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
    figs["plus"], figs["minus"] = fig_a, fig_b

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







class Stream:
    def __init__(self, fb=-1, save_imgs=False):
        self.save_imgs = save_imgs
        self.a_frames = list()
        self.b_frames = list()
        self.b_prime_frames = list()

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

        self.static_center_a = None
        self.static_center_b = None

        self.static_sigmas_x = None
        self.static_sigmas_y = None

        self.roi_a = None
        self.roi_b = None



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




    def get_cameras(self, config_files):
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


    def grab_frames(self, warp_matrix=None):
        try:
            grab_result_a = self.cam_a.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            grab_result_b = self.cam_b.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
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
            raise e

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
        img_a = cv2.circle(a_16bit_color, center_a, 10, (0, 255, 0), 2)
        img_b = cv2.circle(b_16bit_color, center_b, 10, (0, 255, 0), 2)
        return img_a, img_b

    def full_img_w_roi_borders(self, img_12bit, center_):

        try:
            mu_x, sigma_x, amp_x = fgp.get_gaus_boundaries_x(img_12bit, center_)
            mu_y, sigma_y, amp_y = fgp.get_gaus_boundaries_y(img_12bit, center_)
            center_x,  center_y = int(center_[0]), int(center_[1])


            try:

                img_12bit[:, int(center_[0]) + int(sigma_x * 2)] = 4095
                img_12bit[:, int(center_[0]) - int(sigma_x * 2)] = 4095

                img_12bit[int(center_[1]) + int(sigma_y * 2), :] = 4095
                img_12bit[int(center_[1]) - int(sigma_y * 2), :] = 4095


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
        start = input("Step 1 - Stream Raw Camera Feed: Proceed? (y/n): ")

        if (self.histocam_a is None or self.histocam_b is None) and histogram:
            self.histocam_a = histocam.Histocam()
            self.histocam_b = histocam.Histocam()

        self.all_cams.StartGrabbing()

        if start.lower() == 'y':
            continue_stream = True

        while continue_stream:
            self.frame_count += 1
            self.current_frame_a, self.current_frame_b = self.grab_frames()
            self.pre_alignment(histogram)
            continue_stream = self.keep_streaming()

        cv2.destroyAllWindows()

        coregister_ = input("Step 2 - Co-Register with Euclidean Transform: Proceed? (y/n): ")
        warp_ = None

        if coregister_.lower() == "y":
            continue_stream = True
            a_8bit = bdc.to_8_bit(self.current_frame_a)
            b_8bit = bdc.to_8_bit(self.current_frame_b)
            warp_ = ic.get_euclidean_transform_matrix(a_8bit, b_8bit)

            print("Warp Matrix Below:\n\n{}\n".format(warp_))
            a = warp_[0][0]
            b = warp_[0][1]
            tx = warp_[0][2]
            c = warp_[1][0]
            d = warp_[1][1]
            ty = warp_[1][2]

            print("\tTranslation X:{}".format(tx))
            print("\tTranslation Y:{}\n".format(ty))

            scale_x = np.sign(a) * (np.sqrt(a ** 2 + b ** 2))
            scale_y = np.sign(d) * (np.sqrt(c ** 2 + d ** 2))

            print("\tScale X:{}".format(scale_x))
            print("\tScale Y:{}\n".format(scale_y))

            phi = np.arctan2(-1.0 * b, a)
            print("\tPhi Y (rad):{}".format(phi))
            print("\tPhi Y (deg):{}\n".format(np.degrees(phi)))

        while continue_stream:
            self.frame_count += 1
            self.current_frame_a, self.current_frame_b = self.grab_frames(warp_matrix=warp_)
            a_as_16bit = bdc.to_16_bit(self.current_frame_a)
            b_as_16bit = bdc.to_16_bit(self.current_frame_b)
            cv2.imshow("A", a_as_16bit)
            cv2.imshow("B Prime", b_as_16bit)
            continue_stream = self.keep_streaming()

        cv2.destroyAllWindows()

        find_centers_ = input("Step 3 - Find Brightest Pixel Locations: Proceed? (y/n): ")

        if find_centers_.lower() == "y":
            continue_stream = True

        while continue_stream:
            self.frame_count += 1
            self.current_frame_a, self.current_frame_b = self.grab_frames(warp_matrix=warp_)

            a_as_16bit = bdc.to_16_bit(self.current_frame_a)
            b_as_16bit = bdc.to_16_bit(self.current_frame_b)
            max_pixel_a, max_pixel_b = self.find_centers(a_as_16bit, b_as_16bit)

            a_as_16bit = cv2.circle(a_as_16bit, max_pixel_a, 10, (0, 255, 0), 2)
            b_as_16bit = cv2.circle(b_as_16bit, max_pixel_b, 10, (0, 255, 0), 2)


            cv2.imshow("A", a_as_16bit)
            cv2.imshow("B Prime", b_as_16bit)
            continue_stream = self.keep_streaming()

        cv2.destroyAllWindows()


        set_centers_ = input("Step 4 - Set Gaussian-Based Static Centers: Proceed? (y/n): ")

        if set_centers_.lower() == "y":
            continue_stream = True
            a_as_16bit = bdc.to_16_bit(self.current_frame_a)
            b_as_16bit = bdc.to_16_bit(self.current_frame_b)

            max_pixel_a, max_pixel_b = self.find_centers(a_as_16bit, b_as_16bit)


            mu_a_x, sigma_a_x, amp_a_x = fgp.get_gaus_boundaries_x(self.current_frame_a, max_pixel_a)
            mu_a_y, sigma_a_y, amp_a_y = fgp.get_gaus_boundaries_y(self.current_frame_a, max_pixel_a)

            mu_b_x, sigma_b_x, amp_b_x = fgp.get_gaus_boundaries_x(self.current_frame_b, max_pixel_b)
            mu_b_y, sigma_b_y, amp_b_y = fgp.get_gaus_boundaries_y(self.current_frame_b, max_pixel_b)

            self.static_center_a = (int(mu_a_x), int(mu_a_y))
            self.static_center_b = (int(mu_b_x), int(mu_b_y))

        while continue_stream:
            self.frame_count += 1
            self.current_frame_a, self.current_frame_b = self.grab_frames(warp_matrix=warp_)
            a_as_16bit = bdc.to_16_bit(self.current_frame_a)
            b_as_16bit = bdc.to_16_bit(self.current_frame_b)
            a_as_16bit = cv2.circle(a_as_16bit, self.static_center_a, 10, (0, 255, 0), 2)
            b_as_16bit = cv2.circle(b_as_16bit, self.static_center_b, 10, (0, 255, 0), 2)
            cv2.imshow("A", a_as_16bit)
            cv2.imshow("B Prime", b_as_16bit)
            continue_stream = self.keep_streaming()

        cv2.destroyAllWindows()

        find_rois_ = input("Step 5 - Find Regions of Interest: Proceed? (y/n): ")

        if find_rois_.lower() == "y":
            continue_stream = True

        while continue_stream:
            self.frame_count += 1
            self.current_frame_a, self.current_frame_b = self.grab_frames(warp_matrix=warp_)
            sigma_x_a = 0
            sigma_y_a = 0

            sigma_x_b = 0
            sigma_y_b = 0

            try:
                for img_12bit in [self.current_frame_a]:
                    center_ = self.static_center_a

                    mu_x, sigma_x_a, amp_x = fgp.get_gaus_boundaries_x(img_12bit, center_)
                    mu_y, sigma_y_a, amp_y = fgp.get_gaus_boundaries_y(img_12bit, center_)

                    img_12bit[:, int(center_[0]) + int(sigma_x_a * 2)] = 4095
                    img_12bit[:, int(center_[0]) - int(sigma_x_a * 2)] = 4095

                    img_12bit[int(center_[1]) + int(sigma_y_a * 2), :] = 4095
                    img_12bit[int(center_[1]) - int(sigma_y_a * 2), :] = 4095



                    if self.frame_count % 10 == 0:
                        print("\tA  - Sigma X, Sigma Y - {}".format((int(sigma_x_a), int(sigma_y_a))))


                for img_12bit in [self.current_frame_b]:
                    center_ = self.static_center_b

                    mu_x, sigma_x_b, amp_x = fgp.get_gaus_boundaries_x(img_12bit, center_)
                    mu_y, sigma_y_b, amp_y = fgp.get_gaus_boundaries_y(img_12bit, center_)

                    img_12bit[:, int(center_[0]) + int(sigma_x_b * 2)] = 4095
                    img_12bit[:, int(center_[0]) - int(sigma_x_b * 2)] = 4095

                    img_12bit[int(center_[1]) + int(sigma_y_b * 2), :] = 4095
                    img_12bit[int(center_[1]) - int(sigma_y_b * 2), :] = 4095

                    a_as_16bit = bdc.to_16_bit(self.current_frame_a)
                    b_as_16bit = bdc.to_16_bit(self.current_frame_b)


                cv2.imshow("A", a_as_16bit)
                cv2.imshow("B Prime", b_as_16bit)

            except Exception:
                print("Exception Occurred")
                pass

            continue_stream = self.keep_streaming()

            if continue_stream is False:
                self.static_sigmas_x = int(max(sigma_a_x, sigma_b_x))
                self.static_sigmas_y = int(max(sigma_a_y, sigma_b_y))

        cv2.destroyAllWindows()

        close_in = input("Step 6 - Close in on ROI: Proceed? (y/n): ")

        if close_in.lower() == "y":
            continue_stream = True

        while continue_stream:
            self.frame_count += 1
            self.current_frame_a, self.current_frame_b = self.grab_frames(warp_matrix=warp_)

            x_a, y_a = self.static_center_a
            x_b, y_b = self.static_center_b

            n_sigma = 2

            self.roi_a = self.current_frame_a[
                         y_a - n_sigma * self.static_sigmas_y: y_a + n_sigma * self.static_sigmas_y + 1,
                         x_a - n_sigma*self.static_sigmas_x: x_a + n_sigma*self.static_sigmas_x + 1
                         ]

            self.roi_b = self.current_frame_b[
                         y_b - n_sigma * self.static_sigmas_y: y_b + n_sigma * self.static_sigmas_y + 1,
                         x_b - n_sigma*self.static_sigmas_x: x_b + n_sigma*self.static_sigmas_x + 1
                         ]

            cv2.imshow("ROI A", bdc.to_16_bit(self.roi_a))
            cv2.imshow("ROI B Prime", bdc.to_16_bit(self.roi_b))
            continue_stream = self.keep_streaming()


        cv2.destroyAllWindows()

        start_algebra = input("Step 7 - Commence Image Algebra: Proceed? (y/n): ")
        figs, histograms, lines = initialize_histograms_rois()
        figs_alg, histograms_alg, lines_alg = initialize_histograms_algebra()

        if start_algebra.lower() == "y":
            continue_stream = True

            #figs["a"].canvas.draw()  # Draw updates subplots in interactive mode
            #figs["b"].canvas.draw()  # Draw updates subplots in interactive mode
            #plt.show()

        while continue_stream:
            self.frame_count += 1
            self.current_frame_a, self.current_frame_b = self.grab_frames(warp_matrix=warp_)


            x_a, y_a = self.static_center_a
            x_b, y_b = self.static_center_b

            n_sigma = 2

            self.roi_a = self.current_frame_a[
                         y_a - n_sigma * self.static_sigmas_y: y_a + n_sigma * self.static_sigmas_y + 1,
                         x_a - n_sigma*self.static_sigmas_x: x_a + n_sigma*self.static_sigmas_x + 1
                         ]

            self.roi_b = self.current_frame_b[
                         y_b - n_sigma * self.static_sigmas_y: y_b + n_sigma * self.static_sigmas_y + 1,
                         x_b - n_sigma *self.static_sigmas_x: x_b + n_sigma*self.static_sigmas_x + 1
                         ]

            h = self.roi_a.shape[0]
            w = self.roi_a.shape[1]

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
            cv2.imshow("ROIs", A_ON_B)


            #cv2.imshow("ROI A", ROI_A_WITH_HISTOGRAM)
            #cv2.imshow("ROI B Prime", ROI_B_WITH_HISTOGRAM)


            plus = cv2.add(self.roi_a, self.roi_b)*16
            minus = cv2.subtract(self.roi_a, self.roi_b)*16

            update_histogram(histograms_alg, lines_alg, "plus", 4096, plus)
            update_histogram(histograms_alg, lines_alg, "minus", 4096, minus)

            figs_alg["plus"].canvas.draw()  # Draw updates subplots in interactive mode
            figs_alg["minus"].canvas.draw()  # Draw updates subplots in interactive mode
            hist_img_plus = np.fromstring(figs_alg["plus"].canvas.tostring_rgb(), dtype=np.uint8, sep='')
            hist_img_minus = np.fromstring(figs_alg["minus"].canvas.tostring_rgb(), dtype=np.uint8, sep='')  # convert  to image
            hist_img_plus = hist_img_plus.reshape(figs_alg["plus"].canvas.get_width_height()[::-1] + (3,))
            hist_img_minus = hist_img_minus.reshape(figs_alg["minus"].canvas.get_width_height()[::-1] + (3,))
            hist_img_plus = cv2.resize(hist_img_plus, (w, h), interpolation=cv2.INTER_AREA)
            hist_img_minus = cv2.resize(hist_img_minus, (w, h), interpolation=cv2.INTER_AREA)
            hist_img_plus = bdc.to_16_bit(cv2.resize(hist_img_plus, (w, h), interpolation=cv2.INTER_AREA), 8)
            hist_img_minus = bdc.to_16_bit(cv2.resize(hist_img_minus, (w, h), interpolation=cv2.INTER_AREA), 8)
            PLUS_WITH_HISTOGRAM = np.concatenate((cv2.cvtColor(hist_img_plus, cv2.COLOR_RGB2BGR), cv2.cvtColor(plus, cv2.COLOR_GRAY2BGR)), axis=1)
            MINUS_WITH_HISTOGRAM = np.concatenate((cv2.cvtColor(hist_img_minus, cv2.COLOR_RGB2BGR), cv2.cvtColor(minus, cv2.COLOR_GRAY2BGR)), axis=1)

            """
            """

            ALGEBRA = np.concatenate((PLUS_WITH_HISTOGRAM, MINUS_WITH_HISTOGRAM), axis=0)
            cv2.imshow("ALGEBRA", ALGEBRA)

            #cv2.imshow("A PLUS B PRIME", PLUS_WITH_HISTOGRAM)
            #cv2.imshow("A MINUS B PRIME", MINUS_WITH_HISTOGRAM)

            #cv2.imshow("ROI B Prime", ROI_B_WITH_HISTOGRAM)

            #cv2.imshow("Add", cv2.add(self.roi_a, self.roi_b)*16)
            #cv2.imshow("Subtract", cv2.subtract(self.roi_a, self.roi_b)*16)

            #print("Shape ROI A: {}".format(self.roi_a.shape))
            #print("Shape ROI B': {}\n\n".format(self.roi_b.shape))


            #print("Shape Hist Image A: {}".format(hist_img_a.shape))
            #print("Shape Hist Image B: {}".format(hist_img_b.shape))


            #hist_img_a = cv2.cvtColor(hist_img_a, cv2.COLOR_RGB2BGR)  # img is rgb, convert to opencv's default bgr
            #hist_img_b = cv2.cvtColor(hist_img_b, cv2.COLOR_RGB2BGR)  # img is rgb, convert to opencv's default bgr



            #cv2.imshow("Hist Cam A", hist_img_a)
            #cv2.imshow("Hist Cam B", hist_img_b)

            #print("hist_img_a\n", hist_img_a)

            #print("hist_img_b\n", hist_img_b)

            continue_stream = self.keep_streaming()

        self.all_cams.StopGrabbing()
