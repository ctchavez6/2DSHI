"""Driver class - you should be able to run main from the command line along with the some command line arguments of
  your choice, and this script will call all other modules/scripts and feed those args/params as needed."""
import sys
import argparse
import os
import stream_cameras_and_histograms as streams
from experiment_set_up import update_camera_configuration as ucc
from experiment_set_up import request_experiment_parameters
from experiment_set_up import write_experimental_params_to_file
from experiment_set_up import get_command_line_parameters
from experiment_set_up import find_previous_run
from experiment_set_up import config_file_setup as cam_setup

import path_management.directory_management as dirs
from stream_tools import stream_tools
from experiment_set_up import user_input_validation as uiv

from datetime import datetime
# Create an instance of an ArgumentParser Object
parser = argparse.ArgumentParser()


if __name__ == "__main__":
    args = None
    run_mode = uiv.determine_run_mode(sys.argv[:])

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

    current_datetime = datetime.now().strftime("%Y_%m_%d__%H_%M")

    print("\nAll Experimental Data will be saved in the following directory:\n\tD:\\{}\n".format(current_datetime))
    print("\nStarting Run: {}\n".format(current_datetime))

    config_file_parameters = ["ExposureTime", "AcquisitionFrameRate"]
    parameter_dictionary = ucc.reduce_dictionary(args, config_file_parameters)

    camera_configurations_folder = os.path.join(os.getcwd(), "camera_configuration_files")
    config_files_by_cam = cam_setup.assign_config_files(parameter_dictionary, args, camera_configurations_folder)

    devices_found, tlFactory_found = streams.find_devices()

    try_module = True

    if try_module:
        stream = stream_tools.Stream()
        stream.get_cameras(config_files_by_cam)
        stream.start(histogram=True)

    if not try_module:
        cameras = streams.get_cameras(
            devices=devices_found,
            tlFactory=tlFactory_found,
            config_files=config_files_by_cam,
            num_cameras=2)

        figs, histograms, lines = streams.initialize_histograms()
        streams.stream_cam_to_histograms(
                cams_dict=cameras,
                figures=figs,
                histograms_dict=histograms,
                lines=lines,
                frame_break=args["FrameBreak"],
                save_imgs=args["SaveImages"],
                save_vids=args["SaveVideos"],
                display_live_histocam=args["DisplayHistocam"],
                save_histocam_reps=args["SaveHistocam"],
                show_raw_data=args["Raw"],
                resize_factor=args["ResizeFactor"],
                grid=args["Grid"])
