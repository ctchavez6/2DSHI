from pypylon import genicam, pylon  # Import relevant pypylon packages/modules
import traceback  # exception handling
import cv2
from image_processing import bit_depth_conversion as bdc
from image_processing import stack_images as stack
from histocam import histocam
from coregistration import img_characterization as ic
from coregistration import find_gaussian_profile as fgp
import os
import numpy as np
import sys



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


class Stream:
    def __init__(self, fb=-1, save_imgs=False):
        self.save_imgs = save_imgs
        self.a_frames = list()
        self.b_frames = list()
        self.cam_a = None
        self.cam_b = None
        self.all_cams = None
        self.latest_grab_results = {"a": None, "b": None}
        self.frame_count = 0
        self.frame_break = fb
        self.break_key = 'q'
        self.coregistration_break_key = 'c'
        self.keypoints_break_key = 'k'
        self.current_frame_a = None
        self.current_frame_b = None
        self.histocam_a = None
        self.histocam_b = None
        self.stacked_streams = None
        self.data_directory = None

    def get_12bit_a_frames(self):
        return self.a_frames

    def get_12bit_b_frames(self):
        return self.b_frames

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

        #print("Gassian Center of Frame A: ", x_a, y_a)
        #print("Gassian Center of Frame B: ", x_b, y_b)

        return (x_a, y_a), (x_b, y_b)




    def grab_frames(self):
        try:
            grab_result_a = self.cam_a.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            grab_result_b = self.cam_b.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grab_result_a.GrabSucceeded() and grab_result_b.GrabSucceeded():
                a, b = grab_result_a.GetArray(), grab_result_b.GetArray()
                if self.save_imgs:
                    self.a_frames.append(a)
                    self.b_frames.append(b)
                return a, b
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
        #print("Calculating x")
        mean_x, stdev_x, amplitude_x = fgp.get_gaus_boundaries_x(img_12bit, center_)
        #print("\nCalculating y")
        #mean_y, stdev_y, amplitude_y = fgp.get_gaus_boundaries_y(img_12bit, center_)

        # Guassian: 5 Sigma X Direction
        x_max, y_max = center_
        img_12bit[:, x_max + int(stdev_x * 5)] = 4095
        img_12bit[:, x_max - int(stdev_x * 5)] = 4095

        # Guassian: 5 Sigma y Direction
        #img_12bit[y_max + int(stdev_y * 5), :] = 4095
        #img_12bit[y_max - int(stdev_y * 5), :] = 4095

        return img_12bit

    def pre_alignment(self, histogram=False, centers=False, roi_borders=False, crop=False):
        a, b = self.current_frame_a, self.current_frame_b

        if roi_borders:
            a_as_16bit = bdc.to_16_bit(a)
            b_as_16bit = bdc.to_16_bit(b)
            ca, cb = self.find_centers(a_as_16bit, b_as_16bit)
            a = self.full_img_w_roi_borders(a, ca)
            b = self.full_img_w_roi_borders(b, cb)

        if histogram:
            self.histocam_a.update(a)
            self.histocam_b.update(b)
            histocams = add_histogram_representations(self.histocam_a.get_figure(),
                                                      self.histocam_b.get_figure(),
                                                      a,
                                                      b)
            cv2.imshow("Cameras with Histograms", histocams)

        else:
            if roi_borders or crop:
                self.show_16bit_representations(a, b, False, False)
            else:
                self.show_16bit_representations(a, b, False, centers)

    def post_alignment(self, histogram=False, homography=None):

        if histogram:
            self.histocam_a.update(self.current_frame_a)
            self.histocam_b.update(self.current_frame_b)
            histocams = add_histogram_representations(self.histocam_a.get_figure(),
                                                      self.histocam_b.get_figure(),
                                                      self.current_frame_a,
                                                      self.current_frame_b)
            cv2.imshow("Cameras with Histograms", histocams)

        else:
            if homography is not None:
                img_b_prime = ic.transform_img(self.current_frame_b, homography)
                self.show_16bit_representations(self.current_frame_a, img_b_prime, b_prime=True)
            else:
                self.show_16bit_representations(self.current_frame_a, self.current_frame_b)

    def start(self, histogram=False):

        if (self.histocam_a is None or self.histocam_b is None) and histogram:
            self.histocam_a = histocam.Histocam()
            self.histocam_b = histocam.Histocam()

        self.all_cams.StartGrabbing()
        continue_stream = True

        while continue_stream:
            self.frame_count += 1
            self.current_frame_a, self.current_frame_b = self.grab_frames()
            self.pre_alignment(histogram)
            continue_stream = self.keep_streaming()

        cv2.destroyAllWindows()

        find_centers_ = input("Step 1 - Find Centers: Proceed? (y/n): ")

        if find_centers_.lower() == "y":
            continue_stream = True
        elif find_centers_.lower() == "n":
            continue_stream = False

        while continue_stream:
            self.frame_count += 1
            self.current_frame_a, self.current_frame_b = self.grab_frames()
            self.pre_alignment(histogram, True)
            continue_stream = self.keep_streaming()

        cv2.destroyAllWindows()

        find_rois_ = input("Step 2 - Find Regions of Interest: Proceed? (y/n): ")

        if find_rois_.lower() == "y":
            print("FINDING ROIs")
            continue_stream = True
        elif find_rois_.lower() == "n":
            self.all_cams.StopGrabbing()
            continue_stream = False

        while continue_stream:
            self.frame_count += 1
            self.current_frame_a, self.current_frame_b = self.grab_frames()
            self.pre_alignment(histogram, True, True)
            continue_stream = self.keep_streaming()

        cv2.destroyAllWindows()




        find_keypoints = input("Attempt to characterize last Images? (y/n): ")
        if find_keypoints.lower() == "n":
            self.all_cams.StopGrabbing()
            continue_stream = False

        satisfaction = "n"
        homography_ = None

        if find_keypoints.lower() == "y":
            while homography_ is None or satisfaction.lower() == "n":

                img_a_8bit = bdc.to_8_bit(self.current_frame_a)
                img_b_8bit = bdc.to_8_bit(self.current_frame_b)

                homography_ = ic.derive_homography(img_a_8bit, img_b_8bit)

                rows, cols = img_b_8bit.shape
                M, inliers = ic.derive_euclidean_transform(img_b_8bit, img_a_8bit)
                print("Euclidean Transform Matrix below")
                print(M)

                homography_components = ic.get_homography_components(homography_)
                translation = homography_components[0]
                angle = homography_components[1]
                scale = homography_components[2]
                shear = homography_components[3]

                #translation_matrix_ = np.float32([[1, 0, 100], [0, 1, 50]])

                #print("Suggested Angle of Rotation: {}".format(angle))
                #print("Suggested translation: {}".format(translation))
                #print("Suggested scale: {}".format(scale))
                #print("Suggested shear: {}".format(shear))




                satisfaction = input("Are you satisfied with the suggestions? (y/n): ")

                if satisfaction == 'y':
                    b_prime = ic.transform_img(self.current_frame_b * 16, homography_)

                    cv2.imshow("B Prime", b_prime)
                    cv2.waitKey(10000)
                    cv2.destroyAllWindows()

                    self.frame_count += 1
                    self.current_frame_a, self.current_frame_b = self.grab_frames()
                    homography_ = None

                    satisfaction = input("Are you satisfied with B-Prime? (y/n): ")

                if satisfaction == 'q':
                    break

        """
        continue_stream = True

        if homography_ is None:
            print("No Homography: this is a A & B stream (No B-Prime)")

        while continue_stream:

            self.frame_count += 1
            self.current_frame_a, self.current_frame_b = self.grab_frames()
            self.post_alignment(histogram, homography_)
            continue_stream = self.keep_streaming()
        
        
        """

        self.all_cams.StopGrabbing()

