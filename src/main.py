"""Driver class - you should be able to run main from the command line along with the some command line arguments of
  your choice, and this script will call all other modules/scripts and feed those args/params as needed."""
import sys
import argparse
import os
import stream_cameras_and_histograms as streams
import update_camera_configuration as ucc
from data_doc import request_experiment_parameters
from data_doc import write_experimental_params_to_file
from data_doc import get_command_line_parameters
from data_doc import find_previous_run
import path_management.directory_management as dirs

from datetime import datetime
# Create an instance of an ArgumentParser Object
parser = argparse.ArgumentParser()



def validate_video_file(video_file_path_string):
    """
    Validates that the input video file exists and is in one of the accepted formats.

    Args:
        video_file_path_string: Path to video file.
    Raises:
        Exception: If invalid file or video.
    """

    if not os.path.exists(video_file_path_string):
        parser.error("\n\tThe file %s does not exist:" % video_file_path_string)

    acceptable_video_extensions = \
        [
            ".mov",
            ".avi",
            ".mp4"
        ]

    filename, file_extension = os.path.splitext(video_file_path_string)

    if file_extension not in acceptable_video_extensions:
        parser.error("\n\n\tPlease use one of the following video formats:" +
                     ''.join("\n\t\t%s" % x for x in acceptable_video_extensions) +
                     "\n\n\tYour input: %s\n" % video_file_path_string
                     )


if __name__ == "__main__":
    adjusted_command_line_input = [arg for arg in sys.argv[:] if not arg.endswith('main.py')]

    parser = get_command_line_parameters.initialize_arg_parser()
    args = vars(parser.parse_args())  # Parse user arguments into a dictionary

    prev_run = find_previous_run.get_latest_run(args)
    run_mode = 0

    user_input_params_other = {
        "Crystal_1_Temp": None,
        "Crystal_2_Temp": None,
        "Target": None,
        "Compensator Angle": None
    }


    if '-rpp' in adjusted_command_line_input:
        run_mode = 1

    if len(adjusted_command_line_input) < 1 or '-rpp' not in adjusted_command_line_input:
        if len(prev_run) >= 1 and dirs.get_latest_run() is not None:
            print("\n")

            prev_run_name = "Previous Run: " + dirs.get_latest_run()[2:]
            print(prev_run_name)
            underline = ""
            for x in range(len(prev_run_name)):
                underline += "="
            print(underline)

            for key in prev_run:
                print("{}: {}".format(key, prev_run[key]))

            prompt = "\nNo experimental parameters entered.\n\nWould you like to implement last run's parameters?\n"

            prompt += "1: Yes\n"
            prompt += "2: No\n"
            prompt += "3: Yes, but with some changes.\n\n"

            run_mode = int(input(prompt))

        else:
            print("No previous run detected. Please input some experimental parameters.")
            crystal_one_temp = request_experiment_parameters.get_crystal_temperature(1)
            crystal_two_temp = request_experiment_parameters.get_crystal_temperature(2)
            target_descriptor = request_experiment_parameters.get_target_description()
            compensator_angle = request_experiment_parameters.get_compensator_angle()

            user_input_params = {
                "Crystal_1_Temp": str(crystal_one_temp),
                "Crystal_2_Temp": str(crystal_two_temp),
                "Target": str(target_descriptor),
                "Compensator Angle": str(compensator_angle)
            }

            run_mode = 4

    if run_mode == 2 or run_mode not in range(1, 5):
        print("\nPlease run again with the appropriate command line parameters.\n")
        sys.exit()

    if run_mode == 1 or run_mode == 4:
        args = {key: prev_run[key] for key in prev_run if key in args.keys()}
        if run_mode == 1:
            user_input_params = {key: prev_run[key] for key in prev_run if key in user_input_params_other.keys()}

    if run_mode == 3:
        string_parameters = ["video_a",
                             "video_b",
                             "camera_configuration_file",
                             "camera_configuration_file_b",
                             "Target"]

        int_parameters = ["SaveImages",
                          "SaveVideos",
                          "FrameBreak",
                          "DisplayHistocam",
                          "SaveHistocam",
                          "Raw",
                          "Grid",
                          "ExposureTime",
                          "AcquisitionFrameRate"]

        float_parameters = ["ResizeFactor",
                            "Crystal_1_Temp",
                            "Crystal_2_Temp",
                            "Compensator Angle"]

        user_parameter_input = input("Please input the parameter you'd like to update or 'q' to exit.\n")

        while user_parameter_input != "q" or user_parameter_input in prev_run.keys():
            user_value_input = input("New value: ")
            if user_parameter_input == "None":
                prev_run[user_parameter_input] = None
            elif user_parameter_input in int_parameters:
                prev_run[user_parameter_input] = int(user_value_input)
            elif user_parameter_input in float_parameters:
                prev_run[user_parameter_input] = float(user_value_input)
            elif user_parameter_input in string_parameters:
                prev_run[user_parameter_input] = user_value_input
            elif prev_run.endswith("RunPreviousParameters"):
                pass

            print("\nUpdated Values:")
            underline = ""
            for x in range(len("Updated Values:")):
                underline += "="
            print(underline)
            for key in prev_run:
                print("{}: {}".format(key, prev_run[key]))

            print("\n")
            user_parameter_input = input("Please input the parameter you'd like to update or 'q' to exit.")
            print("\n")

    user_input_params = {key: prev_run[key] for key in prev_run if key in user_input_params_other.keys()}

    parser = get_command_line_parameters.initialize_arg_parser()
    args = vars(parser.parse_args())  # Parse user arguments into a dictionary

    current_datetime = datetime.now().strftime("%Y_%m_%d__%H_%M")
    print("\nStarting Run: {}\n".format(current_datetime))

    print("All Experimental Data will be saved in the following directory:\n\tD:\\{}\n".format(current_datetime))

    """

    """


    if args["video_a"] is None and args["video_b"] is None:
        print("No file specified: Attempting video stream")
    elif args["video_a"] is None or args["video_b"] is None:
        print("To run a playback of data, you must specify two valid video files.")
    else:
        validate_video_file(args["video_a"])  # Raises error if not a valid video
        validate_video_file(args["video_b"])  # Raises error if not a valid video

    relevant_parameters = ["ExposureTime", "AcquisitionFrameRate"]
    parameter_dictionary = ucc.reduce_dictionary(args, relevant_parameters)

    write_experimental_params_to_file.document_run(args, user_input_params, current_datetime)

    camera_configurations_folder = os.path.join(os.getcwd(), "camera_configuration_files")

    # Case when both Cameras read the same updated default camera configuration file: CCF'
    if len(parameter_dictionary) > 0 and \
            (args["camera_configuration_file"] is None
                and args["camera_configuration_file_a"] is None
                and args["camera_configuration_file_b"] is None):
        ucc.update_camera_configuration(parameter_dictionary)
        camera_a_configuration = os.path.join(camera_configurations_folder, "updated_configuration_file.pfs")
        camera_b_configuration = camera_a_configuration

    # Case when both Cameras read the same alternate camera configuration file: CCF*
    elif args["camera_configuration_file"] is not None \
            and args["camera_configuration_file_a"] is None \
            and args["camera_configuration_file_b"] is None:
        camera_a_configuration = args["camera_configuration_file"]
        camera_b_configuration = args["camera_configuration_file"]

    # Case when Camera A reads an alternate camera configuration file while B Reads Default: CCF_A*
    elif args["camera_configuration_file"] is None \
            and args["camera_configuration_file_a"] is not None \
            and args["camera_configuration_file_b"] is None:
        camera_a_configuration = args["cam_a_default"]
        camera_b_configuration = os.path.join(camera_configurations_folder, "cam_a_default.pfs")

    # Case when Camera A reads an default camera configuration file while B Reads Alternate CCF: CCF_B*
    elif args["camera_configuration_file"] is None \
            and args["camera_configuration_file_a"] is None \
            and args["camera_configuration_file_b"] is not None:
        camera_a_configuration = os.path.join(camera_configurations_folder, "cam_a_default.pfs")
        camera_b_configuration = args["cam_b_default"]
    else:
        camera_a_configuration = os.path.join(camera_configurations_folder, "cam_a_default.pfs")
        camera_b_configuration = os.path.join(camera_configurations_folder, "cam_b_default.pfs")

    config_files_by_cam = {"a": camera_a_configuration, "b": camera_b_configuration}

    devices_found, tlFactory_found = streams.find_devices()

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
