import cv2
import os, sys
import pickle
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import traceback
from src.pypylon import pylon  # Import relevant pypylon packages/modules
from image_processing import bit_depth_conversion as bdc
from coregistration import find_gaussian_profile as fgp
from stream_tools import App as tk_app
from stream_tools import histograms as hgs
from stream_tools import s1, s2, s3, s4, s5, s6, s7
from experiment_set_up import user_input_validation as uiv

twelve_bit_max = (2 ** 12) - 1
eight_bit_max = (2 ** 8) - 1

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
        self.break_key = 'q'
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

        self.R_HIST = None

        self.stats = None
        self.r_frames = None
        self.a_frames = None

        self.a_images = None
        self.b_prime_images = None

        self.start_writing_at = 0
        self.end_writing_at = 0

        self.max_pixel_a = None
        self.max_pixel_b = None


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

    def offer_to_jump(self, highest_possible_jump):
        desc = "Would you like to use the previous parameters to JUMP to a specific step?"
        offer = uiv.yes_no_quit(desc)
        #offer = input("Would you like to use the previous parameters to JUMP to a specific step? (y/n): ")
        level_descriptions = {
            1: "Step 1 : Stream Raw Camera Feed",
            2: "Step 2 : Co-Register with Euclidean Transform",
            3: "Step 3 : Find Brightest Pixel Locations",
            4: "Step 4 : Set Gaussian-Based Static Centers",
            5: "Step 5 : Define Regions of Interest",
            6: "Step 6 : Close in on ROI & Re-Co Register",
            7: "Step 7 : Commence Image Algebra (Free Stream)"
        }

        options = set()

        if offer is True:
            for i in range(1, max(level_descriptions.keys()) + 1):
                if i <= highest_possible_jump:
                    print(level_descriptions[i])
                    options.add(str(i))

            jump_level_input = input("Which level would you like to jump to?  ")

            while not uiv.valid_input(jump_level_input, options):
                jump_level_input = input("Which level would you like to jump to?  ")

            self.jump_level = int(jump_level_input)

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

    def keep_streaming(self, one_by_one=False):
        if cv2.waitKey(1) & 0xFF == ord(self.break_key):
            return False
        if (not one_by_one) and not self.all_cams.IsGrabbing():
            return False
        if one_by_one and not self.all_cams.IsGrabbing():
            return True
        return True

    def find_centers(self, frame_a_16bit, frame_b_16bit):

        x_a, y_a = fgp.get_coordinates_of_maximum(frame_a_16bit)
        x_b, y_b = fgp.get_coordinates_of_maximum(frame_b_16bit)

        return (x_a, y_a), (x_b, y_b)


    def grab_frames(self, warp_matrix=None):
        try:
            timeout_ms = 5000
            grab_result_a = self.cam_a.RetrieveResult(timeout_ms, pylon.TimeoutHandling_ThrowException)
            grab_result_b = self.cam_b.RetrieveResult(timeout_ms, pylon.TimeoutHandling_ThrowException)

            if grab_result_a.GrabSucceeded() and grab_result_b.GrabSucceeded():
                a, b = grab_result_a.GetArray(), grab_result_b.GetArray()
                grab_result_a.Release()
                grab_result_b.Release()
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
        a_as_16bit, b_as_16bit = bdc.to_16_bit(a_as_12bit), bdc.to_16_bit(b_as_12bit)

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

    def show_frame_by_frame(self):
        pass


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
            histocams = hgs.add_histogram_representations(self.histocam_a.get_figure(), self.histocam_b.get_figure(), a, b)
            cv2.imshow("Cameras with Histograms", histocams)
        else:
            if roi_borders or crop:
                self.show_16bit_representations(a, b, False, False)
            else:
                self.show_16bit_representations(a, b, False, centers)

    def start(self, config_files, histogram=False):
        tlFactory = pylon.TlFactory.GetInstance()
        devices = tlFactory.EnumerateDevices()
        if len(devices) == 0:
            raise Exception("No camera present.")

        self.all_cams = pylon.InstantCameraArray(2)
        self.cam_a, self.cam_b = self.all_cams[0], self.all_cams[1]
        for i, camera in enumerate(self.all_cams):
            camera.Attach(tlFactory.CreateDevice(devices[i]))
            camera.Open()
            pylon.FeaturePersistence.Load(config_files[chr(97 + i)], camera.GetNodeMap())

        # Starts grabbing for all cameras
        self.all_cams.StartGrabbing(pylon.GrabStrategy_LatestImageOnly, pylon.GrabLoop_ProvidedByUser)

        continue_stream = False

        if self.jump_level <= 1:
            s1.step_one(self, histogram, continue_stream)
        else:
            self.current_frame_a, self.current_frame_b = self.grab_frames()

        if self.jump_level <= 2:
            s2.step_two(self, continue_stream)
        else:
            s2.step_two(self, continue_stream, autoload_prev_wm1=True)

        if self.jump_level <= 3:
            s3.step_three(self, continue_stream)
        else:
            s3.step_three(self, continue_stream, autoload=True)

        if self.warp_matrix is None:
            self.jump_level = 10

        if self.jump_level <= 4:
            s4.step_four(self)
        else:
            s4.step_four(self, autoload_prev_static_centers=True)

        if self.jump_level <= 5:
            s5.step_five(self, continue_stream)
        else:
            s5.step_five(self, continue_stream, autoload_roi=True)

        if self.jump_level <= 6:
            s6.step_six_a(self, continue_stream)
            #s6.step_six_b(self, continue_stream, app)
            #s6.step_six_c(self, continue_stream)
        #else:
            #s6.load_wm2_if_present(self)
        app = tk_app.App()
        figs, histograms, lines = hgs.initialize_histograms_rois()
        figs_alg, histograms_alg, lines_alg = hgs.initialize_histograms_algebra()
        figs_r, histograms_r, lines_r = hgs.initialize_histograms_r()

        step = 7

        if self.static_center_a is None or self.static_center_b is None:
            print("Regions of Interest not defined: Exiting Program")
            sys.exit(0)

        if self.jump_level <= step:
            s7.step_seven(self, app, figs, histograms, lines, histograms_alg, lines_alg, figs_alg,
                   histograms_r, lines_r, figs_r)

        self.stats = list()
        self.a_frames = list()
        self.b_prime_frames = list()

        self.a_images = list()
        self.b_prime_images = list()
        self.all_cams.StopGrabbing()



#prev_run = find_previous_run.get_latest_run()
#uiv.display_dict_values(prev_run)
#args = prev_run
calibration_params_dir = os.path.join(os.path.join(os.getcwd(), "test"), "calibration_params")

prev_run_path = os.path.join(calibration_params_dir, "prev_run_dict.pickle")

camera_config_files_directory = os.path.join(os.getcwd(), 'camera_configuration_files')
config_file = os.path.join(camera_config_files_directory, 'cam_a_default.pfs')
cfiles = {'a': config_file, 'b': config_file}

with open(prev_run_path, 'rb') as handle:
    args = pickle.load(handle)

# with open('prev_run_dict.pickle', 'wb') as handle:
# pickle.dump(prev_run, handle, protocol=pickle.HIGHEST_PROTOCOL)

t = Stream(fb=args["FrameBreak"], save_imgs=args["SaveImages"])  # Create a Stream() Instance
t.start(config_files=cfiles)

"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from stream_tools import s1, s2, s3, s4, s5, s6, s7, s8, s9, s10

from image_processing import bit_depth_conversion as bdc
from coregistration import find_gaussian_profile as fgp
from stream_tools import App as tk_app
from stream_tools import histograms as hgs
from experiment_set_up import user_input_validation as uiv
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import traceback
from pypylon import pylon  # Import relevant pypylon packages/modules
import cv2

feedback = True
if feedback:
    print("\ns7profile.py script has begun")
y_n_msg = "Proceed? (y/n): "
sixteen_bit_max = (2 ** 16) - 1
twelve_bit_max = (2 ** 12) - 1
eight_bit_max = (2 ** 8) - 1
camera_config_files_directory = os.path.join(os.getcwd(), 'camera_configuration_files')
config_file = os.path.join(camera_config_files_directory, 'cam_a_default.pfs')
cfiles = {'a': config_file, 'b': config_file}

calibration_params_dir = os.path.join(os.path.join(os.getcwd(), "test"), "calibration_params")

wm1_path = os.path.join(calibration_params_dir, "wm1.npy")
static_center_a_path = os.path.join(calibration_params_dir, "static_center_a.p")
static_center_b_path = os.path.join(calibration_params_dir, "static_center_b.p")
static_sigma_x_path = os.path.join(calibration_params_dir, "static_sigma_x.p")
static_sigma_y_path = os.path.join(calibration_params_dir, "static_sigma_y.p")

if feedback:
    print()
    print("Warp Matrix: ",  wm1_path, "\tExists?", os.path.exists(wm1_path))
    print("Static Center A: ",  wm1_path, "\tExists?", os.path.exists(wm1_path))
    print("Static Center B: ",  wm1_path, "\tExists?", os.path.exists(wm1_path))
    print("Static Sigma X: ",  wm1_path, "\tExists?", os.path.exists(wm1_path))
    print("Static Sigma Y: ",  wm1_path, "\tExists?", os.path.exists(wm1_path))
    print()


"""
"""

class Stream:
    def __init__(self, fb=-1, save_imgs=False):
        if feedback:
            print("An instance of a Stream object has been initialized ")
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
        self.break_key = 'q'
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

        self.R_HIST = None

        self.stats = None
        self.r_frames = None
        self.a_frames = None

        self.a_images = None
        self.b_prime_images = None

        self.start_writing_at = 0
        self.end_writing_at = 0

        self.max_pixel_a = None
        self.max_pixel_b = None
        #self.warp_matrix = None #np.load(wm1_path, allow_pickle=True)
        #self.static_center_a = None #np.load(static_center_a_path, allow_pickle=True)
        #self.static_center_b = None #np.load(static_center_b_path, allow_pickle=True)
        #self.static_sigmas_x = None #np.load(static_sigma_x_path, allow_pickle=True)
        #self.static_sigmas_y = None #np.load(static_sigma_y_path, allow_pickle=True)

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

    def offer_to_jump(self, highest_possible_jump):
        desc = "Would you like to use the previous parameters to JUMP to a specific step?"
        offer = uiv.yes_no_quit(desc)
        #offer = input("Would you like to use the previous parameters to JUMP to a specific step? (y/n): ")
        level_descriptions = {
            1: "Step 1 : Stream Raw Camera Feed",
            2: "Step 2 : Co-Register with Euclidean Transform",
            3: "Step 3 : Find Brightest Pixel Locations",
            4: "Step 4 : Set Gaussian-Based Static Centers",
            5: "Step 5 : Define Regions of Interest",
            6: "Step 6 : Close in on ROI & Re-Co Register",
            7: "Step 7 : Commence Image Algebra (Free Stream)"
        }

        options = set()

        if offer is True:
            for i in range(1, max(level_descriptions.keys()) + 1):
                if i <= highest_possible_jump:
                    print(level_descriptions[i])
                    options.add(str(i))

            jump_level_input = input("Which level would you like to jump to?  ")

            while not uiv.valid_input(jump_level_input, options):
                jump_level_input = input("Which level would you like to jump to?  ")

            self.jump_level = int(jump_level_input)

    def get_cameras(self, config_files):
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

    def keep_streaming(self, one_by_one=False):
        if cv2.waitKey(1) & 0xFF == ord(self.break_key):
            return False
        if (not one_by_one) and not self.all_cams.IsGrabbing():
            return False
        if one_by_one and not self.all_cams.IsGrabbing():
            return True
        return True

    def find_centers(self, frame_a_16bit, frame_b_16bit):

        x_a, y_a = fgp.get_coordinates_of_maximum(frame_a_16bit)
        x_b, y_b = fgp.get_coordinates_of_maximum(frame_b_16bit)

        return (x_a, y_a), (x_b, y_b)


    def grab_frames(self, warp_matrix=None):
        try:
            timeout_ms = 5000
            grab_result_a = self.cam_a.RetrieveResult(timeout_ms, pylon.TimeoutHandling_ThrowException)
            grab_result_b = self.cam_b.RetrieveResult(timeout_ms, pylon.TimeoutHandling_ThrowException)

            if grab_result_a.GrabSucceeded() and grab_result_b.GrabSucceeded():
                a, b = grab_result_a.GetArray(), grab_result_b.GetArray()
                grab_result_a.Release()
                grab_result_b.Release()
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
        a_as_16bit, b_as_16bit = bdc.to_16_bit(a_as_12bit), bdc.to_16_bit(b_as_12bit)

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

    def show_frame_by_frame(self):
        pass


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
            histocams = hgs.add_histogram_representations(self.histocam_a.get_figure(), self.histocam_b.get_figure(), a, b)
            cv2.imshow("Cameras with Histograms", histocams)
        else:
            if roi_borders or crop:
                self.show_16bit_representations(a, b, False, False)
            else:
                self.show_16bit_representations(a, b, False, centers)




    def start(self, config_files, histogram=False):
        tlFactory = pylon.TlFactory.GetInstance()
        devices = tlFactory.EnumerateDevices()
        if len(devices) == 0:
            raise Exception("No camera present.")

        self.all_cams = pylon.InstantCameraArray(2)
        self.cam_a, self.cam_b = self.all_cams[0], self.all_cams[1]
        for i, camera in enumerate(self.all_cams):
            camera.Attach(tlFactory.CreateDevice(devices[i]))
            camera.Open()
            pylon.FeaturePersistence.Load(config_files[chr(97 + i)], camera.GetNodeMap())

        # Starts grabbing for all cameras
        if feedback:
            print("Attempting all_cams.StartGrabbing()")
        self.all_cams.StartGrabbing(pylon.GrabStrategy_LatestImageOnly, pylon.GrabLoop_ProvidedByUser)
        # Starts grabbing for all cameras
        if feedback:
            print("all_cams.StartGrabbing() was successful")


        if feedback:
            print("Starting Image Algebra Free Stream")
        continue_stream = False

        if self.jump_level <= 1:
            s1.step_one(self, histogram, continue_stream)
        else:
            self.current_frame_a, self.current_frame_b = self.grab_frames()

        #self.warp_matrix = np.load(wm1_path, allow_pickle=True)
        if self.jump_level <= 2:
            s2.step_two(self, continue_stream)
        else:
            s2.step_two(self, continue_stream, autoload_prev_wm1=True)

        if self.jump_level <= 3:
            s3.step_three(self, continue_stream)
        else:
            s3.step_three(self, continue_stream, autoload=True)

        if self.warp_matrix is None:
            self.jump_level = 10

        if self.jump_level <= 4:
            s4.step_four(self)
        else:
            s4.step_four(self, autoload_prev_static_centers=True)

        if self.jump_level <= 5:
            s5.step_five(self, continue_stream)
        else:
            s5.step_five(self, continue_stream, autoload_roi=True)

        if self.jump_level <= 6:
            s6.step_six_a(self, continue_stream)
            #s6.step_six_b(self, continue_stream, app)
            #s6.step_six_c(self, continue_stream)
        #else:
            #s6.load_wm2_if_present(self)
        app = tk_app.App()
        figs, histograms, lines = hgs.initialize_histograms_rois()
        figs_alg, histograms_alg, lines_alg = hgs.initialize_histograms_algebra()
        figs_r, histograms_r, lines_r = hgs.initialize_histograms_r()

        step = 7
        if self.static_center_a is None or self.static_center_b is None:
            print("Regions of Interest not defined: Exiting Program")
            self.all_cams.StopGrabbing()
            raise Exception("Eh")

        if self.jump_level <= step:
            step_seven(self, app, figs, histograms, lines, histograms_alg, lines_alg, figs_alg,
                   histograms_r, lines_r, figs_r)

        if feedback:
            print("Image Algebra Free Stream has successfully finished.")
        self.all_cams.StopGrabbing()



def step_seven(stream, app, figs, histograms, lines, histograms_alg, lines_alg, figs_alg,
               histograms_r, lines_r, figs_r):

    desc = "Step 7 - Commence Image Algebra (Free Stream):"
    continue_stream = uiv.yes_no_quit(desc)

    while continue_stream:
        stream.frame_count += 1
        stream.current_frame_a, stream.current_frame_b = stream.grab_frames(warp_matrix=stream.warp_matrix)

        x_a, y_a = stream.static_center_a
        x_b, y_b = stream.static_center_b

        n_sigma = 3

        stream.roi_a = stream.current_frame_a[
                     y_a - n_sigma * stream.static_sigmas_y: y_a + n_sigma * stream.static_sigmas_y + n_sigma,
                     x_a - n_sigma * stream.static_sigmas_x: x_a + n_sigma * stream.static_sigmas_x + n_sigma]

        stream.roi_b = stream.current_frame_b[
                     y_b - n_sigma * stream.static_sigmas_y: y_b + n_sigma * stream.static_sigmas_y + n_sigma,
                     x_b - n_sigma * stream.static_sigmas_x: x_b + n_sigma * stream.static_sigmas_x + n_sigma]

        if stream.warp_matrix_2 is None:
            roi_a = stream.roi_a
            b_double_prime = stream.roi_b
        else:
            roi_a, b_double_prime = stream.grab_frames2(stream.roi_a.copy(), stream.roi_b.copy(), stream.warp_matrix_2.copy())

        CENTER_B_DP = int(b_double_prime.shape[1] * 0.5), int(b_double_prime.shape[0] * 0.5)

        x_a, y_a = CENTER_B_DP
        x_b, y_b = CENTER_B_DP
        n_sigma = app.foo

        stream.roi_a = stream.roi_a[
                     int(y_a - n_sigma * stream.static_sigmas_y): int(y_a + n_sigma * stream.static_sigmas_y + 1),
                     int(x_a - n_sigma * stream.static_sigmas_x): int(x_a + n_sigma * stream.static_sigmas_x + 1)]

        b_double_prime = b_double_prime[
                         int(y_b - n_sigma * stream.static_sigmas_y): int(y_b + n_sigma * stream.static_sigmas_y + 1),
                         int(x_b - n_sigma * stream.static_sigmas_x): int(x_b + n_sigma * stream.static_sigmas_x + 1)]

        stream.roi_b = b_double_prime
        h = b_double_prime.shape[0]
        w = b_double_prime.shape[1]

        hgs.update_histogram(histograms, lines, "a", 4096, stream.roi_a)
        hgs.update_histogram(histograms, lines, "b", 4096, stream.roi_b)
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
            (cv2.cvtColor(hist_img_a, cv2.COLOR_RGB2BGR), cv2.cvtColor(stream.roi_a * 16, cv2.COLOR_GRAY2BGR)), axis=1)
        ROI_B_WITH_HISTOGRAM = np.concatenate(
            (cv2.cvtColor(hist_img_b, cv2.COLOR_RGB2BGR), cv2.cvtColor(stream.roi_b * 16, cv2.COLOR_GRAY2BGR)), axis=1)

        A_ON_B = np.concatenate((ROI_A_WITH_HISTOGRAM, ROI_B_WITH_HISTOGRAM), axis=0)

        plus_ = cv2.add(stream.roi_a, stream.roi_b)
        minus_ = np.zeros(stream.roi_a.shape, dtype='int16')
        minus_ = np.add(minus_, stream.roi_a)
        minus_ = np.add(minus_, stream.roi_b * (-1))
        # print("Lowest pixel in the minus spectrum: {}".format(np.min(minus_.flatten())))

        hgs.update_histogram(histograms_alg, lines_alg, "plus", 4096, plus_, plus=True)
        hgs.update_histogram(histograms_alg, lines_alg, "minus", 4096, minus_, minus=True)

        displayable_plus = cv2.add(stream.roi_a, stream.roi_b) * 16
        displayable_minus = cv2.subtract(stream.roi_a, stream.roi_b) * 16

        figs_alg["plus"].canvas.draw()  # Draw updates subplots in interactive mode
        hist_img_plus = np.fromstring(figs_alg["plus"].canvas.tostring_rgb(), dtype=np.uint8, sep='')
        hist_img_plus = hist_img_plus.reshape(figs_alg["plus"].canvas.get_width_height()[::-1] + (3,))
        hist_img_plus = cv2.resize(hist_img_plus, (w, h), interpolation=cv2.INTER_AREA)
        hist_img_plus = bdc.to_16_bit(cv2.resize(hist_img_plus, (w, h), interpolation=cv2.INTER_AREA), 8)
        PLUS_WITH_HISTOGRAM = np.concatenate(
            (cv2.cvtColor(hist_img_plus, cv2.COLOR_RGB2BGR), cv2.cvtColor(displayable_plus, cv2.COLOR_GRAY2BGR)),
            axis=1)

        figs_alg["minus"].canvas.draw()  # Draw updates subplots in interactive mode
        hist_img_minus = np.fromstring(figs_alg["minus"].canvas.tostring_rgb(), dtype=np.uint8,
                                       sep='')  # convert  to image
        hist_img_minus = hist_img_minus.reshape(figs_alg["minus"].canvas.get_width_height()[::-1] + (3,))
        hist_img_minus = cv2.resize(hist_img_minus, (w, h), interpolation=cv2.INTER_AREA)
        hist_img_minus = bdc.to_16_bit(cv2.resize(hist_img_minus, (w, h), interpolation=cv2.INTER_AREA), 8)
        MINUS_WITH_HISTOGRAM = np.concatenate(
            (cv2.cvtColor(hist_img_minus, cv2.COLOR_RGB2BGR), cv2.cvtColor(displayable_minus, cv2.COLOR_GRAY2BGR)),
            axis=1)

        ALGEBRA = np.concatenate((PLUS_WITH_HISTOGRAM, MINUS_WITH_HISTOGRAM), axis=0)
        DASHBOARD = np.concatenate((A_ON_B, ALGEBRA), axis=1)
        dash_height, dash_width, dash_channels = DASHBOARD.shape
        if dash_width > 2000:
            scale_factor = float(float(2000) / float(dash_width))
            DASHBOARD = cv2.resize(DASHBOARD, (int(dash_width * scale_factor), int(dash_height * scale_factor)))
        cv2.imshow("Dashboard", DASHBOARD)

        R_MATRIX = np.divide(minus_, plus_)
        nan_mean = np.nanmean(R_MATRIX.flatten())
        nan_st_dev = np.nanstd(R_MATRIX.flatten())

        DISPLAYABLE_R_MATRIX = np.zeros((R_MATRIX.shape[0], R_MATRIX.shape[1], 3), dtype=np.uint8)
        DISPLAYABLE_R_MATRIX[:, :, 1] = np.where(R_MATRIX < 0.00, abs(R_MATRIX * (2 ** 8 - 1)), 0)
        DISPLAYABLE_R_MATRIX[:, :, 2] = np.where(R_MATRIX < 0.00, abs(R_MATRIX * (2 ** 8 - 1)), 0)

        DISPLAYABLE_R_MATRIX[:, :, 2] = np.where(R_MATRIX > 0.00, abs(R_MATRIX * (2 ** 8 - 1)),
                                                 DISPLAYABLE_R_MATRIX[:, :, 2])

        dr_height, dr_width, dr_channels = DISPLAYABLE_R_MATRIX.shape

        hgs.update_histogram(histograms_r, lines_r, "r", 4096, R_MATRIX, r=True)
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

        # Mean: {0:.4f}\n".format(nan_mean, 2.000*float(stream.frame_count))
        color = 'rgb(0, 0, 0)'  # black color
        draw.text((x, y), message, fill=color, font=font)
        R_VALUES = np.array(R_VALUES)
        VALUES_W_HIST = np.concatenate((R_VALUES * (2 ** 8), np.array(R_HIST)), axis=1)

        cv2.imshow("R_MATRIX",
                   np.concatenate((VALUES_W_HIST, np.array(DISPLAYABLE_R_MATRIX * (2 ** 8), dtype='uint16')), axis=1))

        continue_stream = stream.keep_streaming()

        if not continue_stream:
            if app is not None:
                app.callback()

            cv2.destroyAllWindows()

        stream.R_HIST = R_HIST


test_stream = Stream()

if feedback:
    print("Attempting to pull camera data")
try:
    test_stream.start(config_files=cfiles)
except Exception as inst:
    if test_stream.all_cams.IsGrabbing():
        test_stream.all_cams.StopGrabbing()
    raise inst
    
"""
