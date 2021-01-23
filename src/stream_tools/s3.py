import cv2
from image_processing import bit_depth_conversion as bdc
from experiment_set_up import user_input_validation as uiv
from experiment_set_up import find_previous_run as fpr
import os
import pickle

y_n_msg = "Proceed? (y/n): "
eight_bit_max = (2 ** 8) - 1

def step_three(stream, continue_stream, autoload=False):
    """
    This step draws a black circle around the location of the brightest pixel in each camera.
    This is useful in two ways:
        1: If brightest pixel is not centered, you likely don't have a gaussian distribution and therefore need to
           reconfigure your beams (likely).
        2: If beam profile is gaussian, knowing the exact or approximate locations of the peak can speed up the process
           of find each beam's gaussian profile (center, amplitude, standard deviation) by giving the algorithm an
           initial estimate for the centers.

    Args:
        stream (Stream): An instance of the Stream class currently connected to your cameras.
        continue_stream (bool): TODO: REMOVE


    """
    previous_run_directory = fpr.get_latest_run_direc(path_override=True, path_to_exclude=stream.current_run)

    prev_bpa_path = os.path.join(previous_run_directory, "max_pixel_a.p")
    prev_bpa_exist = os.path.exists(prev_bpa_path)

    prev_bpb_path = os.path.join(previous_run_directory, "max_pixel_b.p")
    prev_bpb_exist = os.path.exists(prev_bpb_path)

    if autoload and prev_bpa_exist and prev_bpb_exist:
        with open(prev_bpa_path, 'rb') as fp:
            stream.max_pixel_a = pickle.load(fp)

        with open(prev_bpb_path, 'rb') as fp:
            stream.max_pixel_b = pickle.load(fp)

        cv2.destroyAllWindows()
        return

    step_description = "Step 3 - Find Brightest Pixel Locations"
    find_centers_ = uiv.yes_no_quit(step_description)

    if find_centers_ is True:
        continue_stream = True

    while continue_stream:
        stream.frame_count += 1
        stream.current_frame_a, stream.current_frame_b = stream.grab_frames(warp_matrix=stream.warp_matrix)

        a_as_16bit = bdc.to_16_bit(stream.current_frame_a)
        b_as_16bit = bdc.to_16_bit(stream.current_frame_b)
        max_pixel_a, max_pixel_b = stream.find_centers(a_as_16bit, b_as_16bit)

        stream.max_pixel_a = max_pixel_a # TODO, REMOVE REDUNDANT VARIABLES
        stream.max_pixel_b = max_pixel_b

        a_as_16bit = cv2.circle(a_as_16bit, max_pixel_a, 10, (0, eight_bit_max, 0), 2)
        b_as_16bit = cv2.circle(b_as_16bit, max_pixel_b, 10, (0, eight_bit_max, 0), 2)

        cv2.imshow("A", a_as_16bit)
        cv2.imshow("B Prime", b_as_16bit)
        continue_stream = stream.keep_streaming()

    cv2.destroyAllWindows()

