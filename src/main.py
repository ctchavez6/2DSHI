"""Driver class - you should be able to run main from the command line along with the some command line arguments of
  your choice, and this script will call all other modules/scripts and feed those args/params as needed."""
import sys
import argparse
import os
import numpy as np

from experiment_set_up import update_camera_configuration as ucc
from experiment_set_up import write_experimental_params_to_file as wptf
from experiment_set_up import get_command_line_parameters
from experiment_set_up import find_previous_run
from experiment_set_up import config_file_setup as cam_setup
from experiment_set_up import user_input_validation as uiv

from stream_tools import stream_tools
from image_processing import bit_depth_conversion as bdc
from path_management import image_management as im
from datetime import datetime

# Create an instance of an ArgumentParser Object
parser = argparse.ArgumentParser()


if __name__ == "__main__":
    args = None
    run_mode = uiv.determine_run_mode(sys.argv[:])
    current_directory = os.getcwd()

    if run_mode == 1:  # Apply the parameters for the previous run, exactly.
        prev_run = find_previous_run.get_latest_run()
        uiv.display_dict_values(prev_run)
        args = prev_run

    if run_mode == 2:  # Apply the parameters specified in the command line, exactly.
        parser = get_command_line_parameters.initialize_arg_parser()
        args = vars(parser.parse_args())  # Parse user arguments into a dictionary)
        uiv.display_dict_values(args)

    if run_mode == 3:  # Apply last runs parameters, but with some modifications you'll specify.
        prev_run = find_previous_run.get_latest_run()
        uiv.display_dict_values(prev_run)
        args = uiv.update_previous_params(prev_run)

    prev_conf = find_previous_run.get_previous_configuration()
    if prev_conf is None:
        print("No Saved Previous Configuration Found")
    else:
        print(prev_conf)
    current_datetime = datetime.now().strftime("%Y_%m_%d__%H_%M")

    run_directory = os.path.join("D:", "\\" + current_datetime)

    if not os.path.exists(run_directory):
        os.mkdir(run_directory)
        os.mkdir(os.path.join(run_directory, "cam_a_frames"))
        os.mkdir(os.path.join(run_directory, "cam_b_frames"))

    wptf.document_run(args, current_datetime)

    print("\nAll Experimental Data will be saved in the following directory:\n\tD:\\{}\n".format(current_datetime))
    print("\nStarting Run: {}\n".format(current_datetime))

    config_file_parameters = ["ExposureTime", "AcquisitionFrameRate"]
    parameter_dictionary = ucc.reduce_dictionary(args, config_file_parameters)

    camera_configurations_folder = os.path.join(os.getcwd(), "camera_configuration_files")
    # Prepare Camera Configuration Files
    config_files_by_cam = cam_setup.assign_config_files(parameter_dictionary, args, camera_configurations_folder)

    stream = stream_tools.Stream(fb=args["FrameBreak"], save_imgs=args["SaveImages"])  # Create a Stream() Instance
    stream.get_cameras(config_files_by_cam)  # Get Basler Cameras, and load corresponding camera configuration files
    stream.set_current_run(current_datetime)
    if prev_conf is not None:
        warp_from_prev_run = np.zeros((2, 3), dtype='float32')
        warp_from_prev_run[0][0] = float(prev_conf['a'])
        warp_from_prev_run[0][1] = float(prev_conf['b'])
        warp_from_prev_run[0][2] = float(prev_conf['tx'])
        warp_from_prev_run[1][0] = float(prev_conf['c'])
        warp_from_prev_run[1][1] = float(prev_conf['d'])
        warp_from_prev_run[1][2] = float(prev_conf['ty'])
        stream.set_warp_matrix(warp_from_prev_run)


        static_a = prev_conf['static_center_a_x'], prev_conf['static_center_a_y']
        static_b = prev_conf['static_center_b_x'], prev_conf['static_center_b_y']
        stream.set_static_centers(static_a, static_b)

        sigma_x, sigma_y = prev_conf['sigma_x'], prev_conf['sigma_y']
        stream.set_static_sigmas(sigma_x, sigma_y)


        warp_from_prev_run2 = np.zeros((2, 3), dtype='float32')
        warp_from_prev_run2[0][0] = float(prev_conf['a2'])
        warp_from_prev_run2[0][1] = float(prev_conf['b2'])
        warp_from_prev_run2[0][2] = float(prev_conf['tx2'])
        warp_from_prev_run2[1][0] = float(prev_conf['c2'])
        warp_from_prev_run2[1][1] = float(prev_conf['d2'])
        warp_from_prev_run2[1][2] = float(prev_conf['ty2'])
        stream.set_warp_matrix2(warp_from_prev_run2)


        stream.offer_to_jump()

    stream.start(histogram=args["DisplayHistocam"])  # Start steam (Display Histogram if user specified so in input)

    print("Stream has ended.")

    if args["SaveImages"]:
        a_frames_dir = os.path.join(run_directory, "cam_a_frames")
        b_frames_dir = os.path.join(run_directory, "cam_b_frames")

        print("Saving 12 bit frames as 16 bit png files.")
        a_frames = stream.get_12bit_a_frames()
        b_frames = stream.get_12bit_b_frames()

        for i in range(len(a_frames)):
            a16 = bdc.to_16_bit(a_frames[i])
            im.save_img("a_{}.png".format(i + 1), a_frames_dir, a16)

        for j in range(len(b_frames)):
            b16 = bdc.to_16_bit(b_frames[j])
            im.save_img("b_{}.png".format(j + 1), b_frames_dir, b16)

    os.chdir(run_directory)
    if stream.get_warp_matrix() is not None and stream.get_warp_matrix2() is not None:
        try:
            print("Writing Stream Configurations to File")
            wptf.document_configurations(
                warp_matrix=stream.get_warp_matrix(),
                sigmas=stream.get_static_sigmas(),
                static_centers=stream.get_static_centers(),
                warp_matrix_2=stream.get_warp_matrix2())
        except TypeError as e:
            print("TYPE ERROR")
            raise e
    print("Done")
    os.chdir(current_directory)
    print("You have completed and exited the script.")



