from pypylon import genicam, pylon  # Import relevant pypylon packages/modules
import cv2
from image_processing import bit_depth_conversion as bdc
from image_processing import stack_images as stack
from histocam import histocam
from coregistration import img_characterization as ic
from coregistration import find_gaussian_profile as fgp
import numpy as np



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
                    #print("Grabbing B Prime")
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

            """
            print("center_x, center_y")
            print(center_x, center_y)
            print("X Direction")
            print("1 Sigma: {}".format(int(sigma_x)))
            print("2 Sigma: {}".format(int(2*sigma_x)))
            print("3 Sigma: {}".format(int(3*sigma_x)))
            print("4 Sigma: {}".format(int(4*sigma_x)))


            print("Y Direction")
            print("1 Sigma: {}".format(int(sigma_y)))
            print("2 Sigma: {}".format(int(2*sigma_y)))
            print("3 Sigma: {}".format(int(3*sigma_y)))
            print("4 Sigma: {}".format(int(4*sigma_y)))

            """

            try:

                img_12bit[:, int(center_[0]) + int(sigma_x * 4)] = 4095
                img_12bit[:, int(center_[0]) - int(sigma_x * 4)] = 4095

                img_12bit[int(center_[1]) + int(sigma_y * 4), :] = 4095
                img_12bit[int(center_[1]) - int(sigma_y * 4), :] = 4095

                """
                x_max, y_max = center_

                print("\tx_max={}".format(x_max))
                print("\tmu_x={}".format(mu_x))
                print("\tsigma_x={}".format(sigma_x))

                print("\ty_max={}".format(y_max))
                print("\tmu_y={}".format(mu_y))
                print("\tsigma_y={}".format(sigma_y))


                print("\t\tSetting x={} to 4095.".format(int(mu_x) + int(sigma_x * 4)))
                print("\t\tSetting x={} to 4095.".format(int(mu_x) - int(sigma_x * 4)))

                print("\t\tSetting y={} to 4095.".format(mu_y + int(sigma_y * 4)))
                print("\t\tSetting y={} to 4095.".format(mu_y - int(sigma_y * 4)))

                
                """

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

        elif coregister_.lower() == "n":
            continue_stream = False

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
        elif find_centers_.lower() == "n":
            continue_stream = False

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


            print("Characterizing A")
            mu_a_x, sigma_a_x, amp_a_x = fgp.get_gaus_boundaries_x(self.current_frame_a, max_pixel_a)
            mu_a_y, sigma_a_y, amp_a_y = fgp.get_gaus_boundaries_y(self.current_frame_a, max_pixel_a)

            print("Characterizing B Prime")
            mu_b_x, sigma_b_x, amp_b_x = fgp.get_gaus_boundaries_x(self.current_frame_b, max_pixel_b)
            mu_b_y, sigma_b_y, amp_b_y = fgp.get_gaus_boundaries_y(self.current_frame_b, max_pixel_b)

            self.static_center_a = (int(mu_a_x), int(mu_a_y))
            self.static_center_b = (int(mu_b_x), int(mu_b_y))

            #self.static_center_a = max_pixel_a
            #self.static_center_b = max_pixel_b

        elif set_centers_.lower() == "n":
            continue_stream = False

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
        print("\tNote: Printing Sigma X, Sigma Y for each Camera every 10 Frames")

        if find_rois_.lower() == "y":
            continue_stream = True
        elif find_rois_.lower() == "n":
            self.all_cams.StopGrabbing()
            continue_stream = False
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

                    img_12bit[:, int(center_[0]) + int(sigma_x_a * 4)] = 4095
                    img_12bit[:, int(center_[0]) - int(sigma_x_a * 4)] = 4095

                    img_12bit[int(center_[1]) + int(sigma_y_a * 4), :] = 4095
                    img_12bit[int(center_[1]) - int(sigma_y_a * 4), :] = 4095



                    if self.frame_count % 10 == 0:
                        print("\tA  - Sigma X, Sigma Y - {}".format((int(sigma_x_a), int(sigma_y_a))))


                for img_12bit in [self.current_frame_b]:
                    center_ = self.static_center_b

                    mu_x, sigma_x_b, amp_x = fgp.get_gaus_boundaries_x(img_12bit, center_)
                    mu_y, sigma_y_b, amp_y = fgp.get_gaus_boundaries_y(img_12bit, center_)

                    img_12bit[:, int(center_[0]) + int(sigma_x_b * 4)] = 4095
                    img_12bit[:, int(center_[0]) - int(sigma_x_b * 4)] = 4095

                    img_12bit[int(center_[1]) + int(sigma_y_b * 4), :] = 4095
                    img_12bit[int(center_[1]) - int(sigma_y_b * 4), :] = 4095

                    a_as_16bit = bdc.to_16_bit(self.current_frame_a)
                    b_as_16bit = bdc.to_16_bit(self.current_frame_b)

                    if self.frame_count % 10 == 0:
                        print("\tB' - Sigma X, Sigma Y - {}".format((int(sigma_x_b), int(sigma_y_b))))

                cv2.imshow("A", a_as_16bit)
                cv2.imshow("B Prime", b_as_16bit)

            except Exception:
                print("Exception Occurred")
                pass

            #self.pre_alignment(histogram, True, True)
            continue_stream = self.keep_streaming()

            if continue_stream is False:
                self.static_sigmas_x = int(max(sigma_a_x, sigma_b_x))
                self.static_sigmas_y = int(max(sigma_a_y, sigma_b_y))

                print("Setting static sigmas:")
                print("self.static_sigmas_x: {}".format(self.static_sigmas_x))
                print("self.static_sigmas_y: {}".format(self.static_sigmas_y))

        cv2.destroyAllWindows()


        close_in = input("Step 6 - Close in on ROI: Proceed? (y/n): ")

        if close_in.lower() == "y":
            continue_stream = True
        elif close_in.lower() == "n":
            continue_stream = False

        while continue_stream:
            self.frame_count += 1
            self.current_frame_a, self.current_frame_b = self.grab_frames(warp_matrix=warp_)

            x_a, y_a = self.static_center_a
            x_b, y_b = self.static_center_b

            self.roi_a = self.current_frame_a[
                         y_a - 4 * self.static_sigmas_y: y_a + 4 * self.static_sigmas_y + 1,
                         x_a - 4*self.static_sigmas_x: x_a + 4*self.static_sigmas_x + 1
                         ]

            self.roi_b = self.current_frame_b[
                         y_b - 4 * self.static_sigmas_y: y_b + 4 * self.static_sigmas_y + 1,
                         x_b - 4*self.static_sigmas_x: x_b + 4*self.static_sigmas_x + 1
                         ]


            roi_a_16bit = bdc.to_16_bit(self.roi_a)
            cv2.imshow("ROI A", roi_a_16bit)

            roi_b_16bit = bdc.to_16_bit(self.roi_b)
            cv2.imshow("ROI B", roi_b_16bit)
            continue_stream = self.keep_streaming()


        """
        if close_in.lower() == "y":


            continue_stream = True
            print("\nClosing in on ROI.")
            print("Stream A:")
            print("\tCenter: {}".format(self.static_center_a))
            x_a, y_a = self.static_center_a
            print("\tX will go from {} to {}".format(x_a - 4*self.static_sigmas_x, x_a + 4*self.static_sigmas_x))
            print("\tY will go from {} to {}".format(y_a - 4*self.static_sigmas_x, y_a + 4*self.static_sigmas_x))


            print("Stream B Prime:")
            print("\tCenter: {}".format(self.static_center_b))
            x_b, y_b = self.static_center_b
            print("\tX will go from {} to {}".format(x_b - 4 * self.static_sigmas_x, x_b + 4 * self.static_sigmas_x))
            print("\tY will go from {} to {}".format(y_b - 4 * self.static_sigmas_y, y_b + 4 * self.static_sigmas_y))

            print("Most Recent Frame")
            self.roi_a = self.current_frame_a[
                         y_a - 4 * self.static_sigmas_y: y_a + 4 * self.static_sigmas_y + 1,
                         x_a - 4*self.static_sigmas_x: x_a + 4*self.static_sigmas_x + 1
                         ]

            self.roi_b = self.current_frame_b[
                         y_b - 4 * self.static_sigmas_y: y_b + 4 * self.static_sigmas_y + 1,
                         x_b - 4*self.static_sigmas_x: x_b + 4*self.static_sigmas_x + 1
                         ]


            print("MOMENT OF TRUTH")
            print("Shape of ROI A : {}".format(self.roi_a.shape))
            print("Shape of ROI B': {}".format(self.roi_b.shape))

                        #self.current_frame_a[
                         #x_a - 4*self.static_sigmas_x: x_a + 4*self.static_sigmas_x + 1,
                         #y_a - 4 * self.static_sigmas_x, y_a + 4 * self.static_sigmas_x
                         #]

            roi_a_16bit = bdc.to_16_bit(self.roi_a)
            cv2.imshow("ROI A", roi_a_16bit)
            cv2.waitKey(10000)

            roi_b_16bit = bdc.to_16_bit(self.roi_b)
            cv2.imshow("ROI B", roi_b_16bit)
            cv2.waitKey(10000)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
            self.frame_count += 1
            self.current_frame_a, self.current_frame_b = self.grab_frames(warp_matrix=warp_)

            x_a, y_a = self.static_center_a



            x_b, y_b = self.static_center_b

            self.roi_a = self.current_frame_a[
                         y_a - 4 * self.static_sigmas_y: y_a + 4 * self.static_sigmas_y + 1,
                         x_a - 4*self.static_sigmas_x: x_a + 4*self.static_sigmas_x + 1
                         ]

            self.roi_b = self.current_frame_b[
                         y_b - 4 * self.static_sigmas_y: y_b + 4 * self.static_sigmas_y + 1,
                         x_b - 4*self.static_sigmas_x: x_b + 4*self.static_sigmas_x + 1
                         ]


            roi_a_16bit = bdc.to_16_bit(self.roi_a)
            cv2.imshow("ROI A", roi_a_16bit)
            #cv2.waitKey(10000)

            roi_b_16bit = bdc.to_16_bit(self.roi_b)
            cv2.imshow("ROI B", roi_b_16bit)
            #cv2.waitKey(10000)


        """



        cv2.destroyAllWindows()



        self.all_cams.StopGrabbing()

