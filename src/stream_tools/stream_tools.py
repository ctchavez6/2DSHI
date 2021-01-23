import traceback
from pypylon import pylon  # Import relevant pypylon packages/modules
import cv2
from image_processing import bit_depth_conversion as bdc
from coregistration import find_gaussian_profile as fgp
import os
from . import App as tk_app
from . import histograms as hgs
from . import s1, s2, s3, s4, s5, s6, s7, s8, s9, s10
from experiment_set_up import user_input_validation as uiv
from . import store_params as sp

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
            histocams = hgs.add_histogram_representations(self.histocam_a.get_figure(), self.histocam_b.get_figure(), a, b)
            cv2.imshow("Cameras with Histograms", histocams)
        else:
            if roi_borders or crop:
                self.show_16bit_representations(a, b, False, False)
            else:
                self.show_16bit_representations(a, b, False, centers)

    def start(self, histogram=False):
        continue_stream = False
        run_folder = os.path.join("D:", "\\" + self.current_run)

        self.all_cams.StartGrabbing()

        if self.jump_level <= 1:
            s1.step_one(self, histogram, continue_stream)
        else:
            self.current_frame_a, self.current_frame_b = self.grab_frames()

        if self.jump_level <= 2:
            s2.step_two(self, continue_stream)
        else:
            s2.step_two(self, continue_stream, autoload_prev_wm1=True)
        sp.store_warp_matrices(self, run_folder)

        if self.jump_level <= 3:
            s3.step_three(self, continue_stream)
        else:
            s3.step_three(self, continue_stream, autoload=True)
        sp.store_brightest_pixels(self, run_folder)

        if self.warp_matrix is None:
            self.jump_level = 10

        step = 4
        if self.jump_level <= step:
            s4.step_four(self)
        else:
            s4.step_four(self, autoload_prev_static_centers=True)
        sp.store_static_centers(self, run_folder)

        step = 5
        if self.jump_level <= step:
            s5.step_five(self, continue_stream)
        else:
            s5.step_five(self, continue_stream, autoload_roi=True)
        sp.store_static_sigmas(self, run_folder)

        app = tk_app.App()
        step = 6

        if self.jump_level <= step:
            s6.step_six_a(self, continue_stream)
            s6.step_six_b(self, continue_stream, app)
            s6.step_six_c(self, continue_stream)
        else:
            s6.load_wm2_if_present(self)
        sp.store_warp_matrices(self, run_folder)

        figs, histograms, lines = hgs.initialize_histograms_rois()
        figs_alg, histograms_alg, lines_alg = hgs.initialize_histograms_algebra()
        figs_r, histograms_r, lines_r = hgs.initialize_histograms_r()

        step = 7

        if self.static_center_a is None or self.static_center_b is None:
            print("Regions of Interest not defined: Exiting Program")
            continue_stream = False

        if self.jump_level <= step:
            start_algebra = input("Step 7 - Commence Image Algebra (Free Stream): Proceed? (y/n): ")
            if start_algebra.lower() == "y":
                continue_stream = True
            s7.step_seven(self, continue_stream, app, figs, histograms, lines, histograms_alg, lines_alg, figs_alg,
                   histograms_r, lines_r, figs_r)

        self.stats = list()
        self.a_frames = list()
        self.b_prime_frames = list()

        self.a_images = list()
        self.b_prime_images = list()


        step = 8
        if not self.jump_level > 8:
            s8.step_eight(self, run_folder, app, figs, histograms, lines, histograms_alg, lines_alg, figs_alg,
               histograms_r, lines_r, figs_r)

        cv2.destroyAllWindows()

        step = 9
        if self.jump_level <= step:
            s9.step_nine(self, self.start_writing_at, self.end_writing_at, run_folder, self.a_images, self.a_frames,
                         self.b_prime_images, self.b_prime_frames, self.stats)


        tk_app.bring_to_front(app)
        tk_app.kill_app(app)

        self.all_cams.StopGrabbing()

        step = 10
        s10.step_ten(run_folder)

        #step = 11
        #s11.step_eleven(self, run_folder)

