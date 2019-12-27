import os

def document_run(all_experimental_parameters, run):
    """
    Takes a dictionary and removes any keys not specified.

    Args:
        parameter_dictionary: Requested number of bins.
    Returns:
        reduced_dictionary: The input dictionary but reduced to the keys specified.
    """

    current_working_directory = os.getcwd()

    run_directory = os.path.join("D:", run)

    if not os.path.exists(run_directory):
        os.mkdir(run_directory)
    os.chdir(run_directory)

    updated_file_as_string = ''

    for key in all_experimental_parameters:
        updated_file_as_string += key
        updated_file_as_string += "\t"
        if all_experimental_parameters[key] is not None:
            updated_file_as_string += str(all_experimental_parameters[key])
        else:
            updated_file_as_string += "None"
        updated_file_as_string += "\n"


    params_file = open('run_parameters.txt', 'w+')
    params_file.write(updated_file_as_string)
    params_file.close()

    os.chdir(current_working_directory)
