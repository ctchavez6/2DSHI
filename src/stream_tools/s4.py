from coregistration import find_gaussian_profile as fgp
from image_processing import bit_depth_conversion as bdc
from experiment_set_up import find_previous_run as fpr
import pickle
import cv2
import os

eight_bit_max = (2 ** 8) - 1
y_n_msg = "Proceed? (y/n): "


def step_four(stream, autoload_prev_static_centers=False):
    previous_run_directory = fpr.get_latest_run_direc(path_override=True, path_to_exclude=stream.current_run)

    prev_sca_path = os.path.join(previous_run_directory, "static_center_a.p")
    prev_sca_exist = os.path.exists(prev_sca_path)

    prev_scb_path = os.path.join(previous_run_directory, "static_center_b.p")
    prev_scb_exist = os.path.exists(prev_scb_path)

    if autoload_prev_static_centers and prev_sca_exist and prev_scb_exist:
        with open(prev_sca_path, 'rb') as fp:
            stream.static_center_a = pickle.load(fp)

        with open(prev_scb_path, 'rb') as fp:
            stream.static_center_b = pickle.load(fp)

        cv2.destroyAllWindows()
        return

    set_centers_ = input("Step 4 - Set Gaussian-Based Static Centers - {}".format(y_n_msg))

    if set_centers_.lower() == "y":
        continue_stream = True
        a_as_16bit = bdc.to_16_bit(stream.current_frame_a)
        b_as_16bit = bdc.to_16_bit(stream.current_frame_b)

        max_pixel_a, max_pixel_b = stream.find_centers(a_as_16bit, b_as_16bit)

        stream.mu_a_x, stream.sigma_a_x, stream.amp_a_x = fgp.get_gaus_boundaries_x(stream.current_frame_a, max_pixel_a)
        stream.mu_a_y, stream.sigma_a_y, stream.amp_a_y = fgp.get_gaus_boundaries_y(stream.current_frame_a, max_pixel_a)

        stream.mu_b_x, stream.sigma_b_x, stream.amp_b_x = fgp.get_gaus_boundaries_x(stream.current_frame_b, max_pixel_b)
        stream.mu_b_y, stream.sigma_b_y, stream.amp_b_y = fgp.get_gaus_boundaries_y(stream.current_frame_b, max_pixel_b)

        print("Setting Centers\n")
        print("Calculated Gaussian Centers")
        stream.static_center_a = (int(stream.mu_a_x), int(stream.mu_a_y))
        # stream.static_center_b = (int(mu_b_x), int(mu_b_y))  # Original
        stream.static_center_b = (int(stream.mu_a_x), int(stream.mu_a_y))  # Picking A

        print("\t\tA                         : {}".format(stream.static_center_a))
        print("\t\tB Prime (Calculated)      : {}".format((int(stream.mu_b_x), int(stream.mu_b_y))))
        print("\t\tB Prime (Overwritten to A): {}".format(stream.static_center_a))

        option = input("Would you like to use the \n\tcalculated gaussian center (y) "
                       "\n\tor the overwritten gaussian center (n):  ")

        if option == "y":
            stream.static_center_b = (int(stream.mu_b_x), int(stream.mu_b_y))  # Picking A
        elif option == "n":
            stream.static_center_b = stream.static_center_a

    else:
        continue_stream = False
        stream.jump_level = 10

    while continue_stream:
        stream.frame_count += 1
        stream.current_frame_a, stream.current_frame_b = stream.grab_frames(warp_matrix=stream.warp_matrix)
        a_as_16bit = bdc.to_16_bit(stream.current_frame_a)
        b_as_16bit = bdc.to_16_bit(stream.current_frame_b)
        a_as_16bit = cv2.circle(a_as_16bit, stream.static_center_a, 10, (0, eight_bit_max, 0), 2)
        b_as_16bit = cv2.circle(b_as_16bit, stream.static_center_b, 10, (0, eight_bit_max, 0), 2)
        cv2.imshow("A", a_as_16bit)
        cv2.imshow("B Prime", b_as_16bit)
        continue_stream = stream.keep_streaming()

    cv2.destroyAllWindows()
