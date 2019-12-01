import os


def reduce_dictionary(input_dictionary, keys_kept):
    """
    Takes a dictionary and removes any keys not specified.

    Args:
        input_dictionary: Requested number of bins.
        keys_kept: Requested number of bins.
    Returns:
        reduced_dictionary: The input dictionary but reduced to the keys specified.
    """

    reduced_dictionary = {key: input_dictionary[key] for key in keys_kept if input_dictionary[key] is not None}
    return reduced_dictionary


def update_camera_configuration(parameter_dictionary):
    print("starting update_camera_configuration")
    current_working_directory = os.getcwd()
    camera_config_files_directory = os.path.join(current_working_directory, "camera_configuration_files")
    default_config_file_path = os.path.join(camera_config_files_directory, "default_camera_configuration.pfs")
    default_config_file = open(default_config_file_path, 'r')
    updated_file_as_string = ''

    line_count = 0
    for line in default_config_file:
        line_count += 1
        if line_count <= 3:
            updated_file_as_string += line
        else:
            split_by_tabs = line.split('\t')
            parameter = split_by_tabs[0]
            value = split_by_tabs[1].rstrip()

            if parameter in parameter_dictionary:
                updated_file_as_string += parameter
                updated_file_as_string += "\t"
                updated_file_as_string += parameter_dictionary[parameter]
                updated_file_as_string += "\n"
            else:
                updated_file_as_string += parameter
                updated_file_as_string += "\t"
                updated_file_as_string += value
                updated_file_as_string += "\n"

    default_config_file.close()

    os.chdir(camera_config_files_directory)

    if os.path.exists((os.path.join(camera_config_files_directory, "updated_configuration_file.pfs"))):
        os.remove(os.path.join(camera_config_files_directory, "updated_configuration_file.pfs"))

    updated_config_file = open('updated_configuration_file.pfs', 'w+')
    updated_config_file.write(updated_file_as_string)
    updated_config_file.close()
    os.chdir(current_working_directory)


