import sys

from . import request_experiment_parameters
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

def update_previous_params():
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
                        "CrystalTemp1",
                        "CrystalTemp2",
                        "CompensatorAngle"]

    print

