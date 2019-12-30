import os
from . import get_command_line_parameters as gclp

def get_latest_run():
    #data_directory = "/Users/ivansepulveda/PycharmProjects/2DSHI/src/tests/D"  # Ivan's Mac
    data_directory = os.path.join("D:", "")  # Windows PC @ Franks' House

    all_runs = sorted([os.path.join(data_directory, path) for path in os.listdir(data_directory)
                       if os.path.isdir(os.path.join(data_directory, path))
                       and path not in ["$RECYCLE.BIN", "System Volume Information"]])

    all_params_dict = dict()

    if len(all_runs) < 1:
        return all_params_dict

    last_run = all_runs[-1]
    last_run_params_file_path = os.path.join(last_run, "run_parameters.txt")

    last_run_params_file = open(last_run_params_file_path, 'r')


    string_parameters = ["video_a",
                         "video_b",
                         "camera_configuration_file",
                         "camera_configuration_file_a"
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



    for line in last_run_params_file:
        split_by_tabs = line.split('\t')
        parameter = split_by_tabs[0]
        value = split_by_tabs[1].rstrip()
        if value.endswith("None"):
            all_params_dict[parameter] = None
        elif parameter in int_parameters:
            all_params_dict[parameter] = int(float(value))
        elif parameter in float_parameters:
            all_params_dict[parameter] = float(value)
        elif parameter in string_parameters:
            all_params_dict[parameter] = value
        elif parameter.endswith("RunPreviousParameters"):
            pass
        else:
            print("Warning: Parameter {} with a value of {} has NOT been accounted for.".format(parameter, value))

    last_run_params_file.close()
    return all_params_dict

def get_latest_run_name(data_directory):
    all_runs = sorted([os.path.join(data_directory, path) for path in os.listdir(data_directory)
                       if os.path.isdir(os.path.join(data_directory, path))
                       and path not in ["$RECYCLE.BIN", "System Volume Information"]])
    if len(all_runs) < 1:
        return ""
    else:
        return all_runs[-1]
