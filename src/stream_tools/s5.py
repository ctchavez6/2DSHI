from coregistration import find_gaussian_profile as fgp
from image_processing import bit_depth_conversion as bdc
import cv2
import traceback
import os
import pickle
from experiment_set_up import find_previous_run as fpr
from experiment_set_up import user_input_validation as uiv
from exceptions import  coregistration_exceptions as cre


def step_five(stream, continue_stream, autoload_roi=False):
    """
    This step finds the Regions of interest of both images.
    The Regions of Interest are comprised of:
        Sigma_X: Standard deviation of beam's gaussian profile in the horizontal direction
        Sigma_Y: Standard deviation of beam's gaussian profile in the vertical direction
        Static Center A: The coordinates of the Gaussian Peak for Camera A
        Static Center B: The coordinates of the Gaussian Peak for Camera B

         _ _ _ _ _ _ _ _ _
        |                 |
        |                 |
        |    (x_a,y_a)    | 2*sigma_y
        |                 |
        |                 |
         _ _ _ _ _ _ _ _ _
             2*sigma_x

    Args:
        stream (Stream): Instance of stream class currently connected to cameras
        continue_stream (bool): Should camera keep streaming. TODO: CHECK IF THIS CAN JUST EXIST IN S5 NAMESPACE

    """
    previous_run_directory = fpr.get_latest_run_direc(path_override=True, path_to_exclude=stream.current_run)

    prev_sigma_x_path = os.path.join(previous_run_directory, "static_sigma_x.p")
    prev_sigma_x_exist = os.path.exists(prev_sigma_x_path)

    prev_sigma_y_path = os.path.join(previous_run_directory, "static_sigma_y.p")
    prev_sigma_y_exist = os.path.exists(prev_sigma_y_path)

    if autoload_roi and prev_sigma_x_exist and prev_sigma_y_exist:
        with open(prev_sigma_x_path, 'rb') as fp:
            stream.static_sigmas_x = pickle.load(fp)

        with open(prev_sigma_y_path, 'rb') as fp:
            stream.static_sigmas_y = pickle.load(fp)

        cv2.destroyAllWindows()
        return

    step_description = "Step 5 - Define Regions of Interest"
    find_rois_ = uiv.yes_no_quit(step_description)

    if find_rois_ is True:
        continue_stream = True

    while continue_stream:
        stream.frame_count += 1
        stream.current_frame_a, stream.current_frame_b = stream.grab_frames(warp_matrix=stream.warp_matrix)

        try:
            for img_12bit in [stream.current_frame_a]:  # TODO: OPTIMIZE, SHOULD NOT BE ITERATING THROUGH LIST OF ONE
                center_ = stream.static_center_a

                n_sigma = 1

                stream.mu_x, stream.sigma_a_x, stream.amp_x = fgp.get_gaus_boundaries_x(img_12bit, center_)
                stream.mu_y, stream.sigma_a_y, stream.amp_y = fgp.get_gaus_boundaries_y(img_12bit, center_)

                img_12bit[:, int(center_[0]) + int(stream.sigma_a_x * n_sigma)] = 4095
                img_12bit[:, int(center_[0]) - int(stream.sigma_a_x * n_sigma)] = 4095

                img_12bit[int(center_[1]) + int(stream.sigma_a_y * n_sigma), :] = 4095
                img_12bit[int(center_[1]) - int(stream.sigma_a_y * n_sigma), :] = 4095

                if stream.frame_count % 15 == 0:
                    print("\tA  - Sigma X, Sigma Y - {}".format((int(stream.sigma_a_x), int(stream.sigma_a_y))))

            for img_12bit in [stream.current_frame_b]:  # TODO: OPTIMIZE, SHOULD NOT BE ITERATING THROUGH LIST OF ONE
                center_ = stream.static_center_b

                stream.mu_x, stream.sigma_b_x, stream.amp_x = fgp.get_gaus_boundaries_x(img_12bit, center_)
                stream.mu_y, stream.sigma_b_y, stream.amp_y = fgp.get_gaus_boundaries_y(img_12bit, center_)

                img_12bit[:, int(center_[0]) + int(stream.sigma_b_x * n_sigma)] = 4095
                img_12bit[:, int(center_[0]) - int(stream.sigma_b_x * n_sigma)] = 4095

                img_12bit[int(center_[1]) + int(stream.sigma_b_y * n_sigma), :] = 4095
                img_12bit[int(center_[1]) - int(stream.sigma_b_y * n_sigma), :] = 4095

                if stream.frame_count % 15 == 0:
                    print("\tB  - Sigma X, Sigma Y - {}".format((int(stream.sigma_b_x), int(stream.sigma_b_y))))

            a_as_16bit = bdc.to_16_bit(stream.current_frame_a)
            b_as_16bit = bdc.to_16_bit(stream.current_frame_b)

            cv2.imshow("A", a_as_16bit)
            cv2.imshow("B Prime", b_as_16bit)

        except IndexError:
            raise cre.InvalidROIException()

        continue_stream = stream.keep_streaming()

        if continue_stream is False:
            stream.static_sigmas_x = int(max(stream.sigma_a_x, stream.sigma_b_x))
            stream.static_sigmas_y = int(max(stream.sigma_a_y, stream.sigma_b_y))

    cv2.destroyAllWindows()