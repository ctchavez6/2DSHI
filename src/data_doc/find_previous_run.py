import os
from data_doc import get_command_line_parameters

def get_latest_run(command_line_args_dict):
    start_directory = os.getcwd()
    data_directory = os.path.join("D:", "")

    all_runs = sorted([os.path.join(data_directory, path) for path in os.listdir(data_directory)
                       if os.path.isdir(os.path.join(data_directory, path))
                       and path not in ["$RECYCLE.BIN", "System Volume Information"]])

    all_params_dict = dict()

    if len(all_runs) < 1:
        return all_params_dict

    last_run = all_runs[-1]
    last_run_params_file_path = os.path.join(last_run, "run_parameters.txt")

    last_run_params_file = open(last_run_params_file_path, 'r')
    command_line_params = command_line_args_dict.keys()


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

    for line in last_run_params_file:
        split_by_tabs = line.split('\t')
        parameter = split_by_tabs[0]
        value = split_by_tabs[1].rstrip()
        #print(parameter, value, type(value))

        if value.endswith("None"):
            all_params_dict[parameter] = None
        elif parameter in int_parameters:
            all_params_dict[parameter] = int(value)
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

def get_latest_run_name():
    data_directory = os.path.join("D:", "")

    all_runs = sorted([os.path.join(data_directory, path) for path in os.listdir(data_directory)
                       if os.path.isdir(os.path.join(data_directory, path))
                       and path not in ["$RECYCLE.BIN", "System Volume Information"]])


    if len(all_runs) < 1:
        return ""
    else:
        return all_runs[-1]