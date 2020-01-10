import sys

from . import write_experimental_params_to_file
from . import get_command_line_parameters
from . import find_previous_run


def display_dict_values(d):
    for key in d:
        print("\t{}: {}".format(key, d[key]))


def determine_run_mode(sys_args):
    run_mode = 0
    adjusted_command_line_input = [arg for arg in sys_args if not arg.endswith('main.py')]
    run_mode_feedback = "Run Mode {}: Run Previous Parameters"


    if '-rpp' in adjusted_command_line_input:
        print(run_mode_feedback.format(1))
        return 1

    if '-rpp' not in adjusted_command_line_input and len(adjusted_command_line_input) > 1:
        prompt = "You've chosen to explicitly enter command line parameters. Would you like to run the experiment" +\
                 "under these exact conditions? (y/n) [Or 'q' to quit.]"
        input_ = input(prompt).lower()
        while input_ not in ["q", "y", "n"]:
            input_ = input(prompt).lower()

        if input_ == "q":
            sys.exit(1)
        elif input_ == "y":
            print(run_mode_feedback.format(2))
            return 2

    if run_mode == 0:
        prompt = "\nNo experimental parameters entered.\n\nWould you like to implement last run's parameters?\n"
        prompt += "a: Yes\n"
        prompt += "b: Yes, but with some changes.\n\n(Or enter 'q' to quit)\n\n"
        input_ = input(prompt).lower()
        options = ["a", "b", "q"]
        while input_ not in options:
            input_ = input(prompt).lower()

        if input_ == "q":
            print("Okay, goodbye.")
            sys.exit(1)
        elif input_ == "a":
            run_mode = 1
            print(run_mode_feedback.format(run_mode))
        elif input_ == "b":
            run_mode = 3
            print(run_mode_feedback.format(run_mode))

    return run_mode

def update_previous_params(args_):

    string_parameters = ["video_a",
                         "video_b",
                         "camera_configuration_file",
                         "camera_configuration_file_a",
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
                        "CrystalTemp1",
                        "CrystalTemp2",
                        "CompensatorAngle"]

    other_params = ["RunPreviousParameters"]

    all_params = string_parameters + int_parameters + float_parameters + other_params

    modified_args = dict(args_).copy()

    for key in modified_args:
        if key not in all_params:
            print("Warning: {} not accounted for.")

    param_prompt = "Enter the parameter you'd like to update (or 'r' to run as is)."
    param_input = input(param_prompt)

    value_prompt = "Enter your new value for {}: "

    while param_input != "r":
        if param_input in all_params:
            new_value = input(value_prompt.format(param_input))

            if param_input in string_parameters:
                modified_args[param_input] = new_value
            elif param_input in int_parameters:
                modified_args[param_input] = float(new_value)
            elif param_input in float_parameters:
                modified_args[param_input] = float(new_value)

            print("Parameters have been updated as shown below.\n")

            for key in modified_args:
                print(key, modified_args[key])

            param_input = input(param_prompt)

        elif param_input not in all_params:
            print("Unfortunately, '{}' is not a valid paramter")
            param_input = input(param_prompt)

    return modified_args
