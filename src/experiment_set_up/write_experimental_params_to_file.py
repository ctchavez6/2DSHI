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
    return


def document_configurations(warp_matrix, sigmas, static_centers, warp_matrix_2):
    """
    Takes a dictionary and removes any keys not specified.

    Args:
        parameter_dictionary: Requested number of bins.
    Returns:
        reduced_dictionary: The input dictionary but reduced to the keys specified.
    """
    current_working_directory = os.getcwd()
    os.chdir(os.path.join("D:"))

    a = warp_matrix[0][0]
    b = warp_matrix[0][1]
    tx = warp_matrix[0][2]
    c = warp_matrix[1][0]
    d = warp_matrix[1][1]
    ty = warp_matrix[1][2]

    updated_file_as_string = ''

    updated_file_as_string += "a"
    updated_file_as_string += "\t"
    updated_file_as_string += str(a)
    updated_file_as_string += "\n"

    updated_file_as_string += "b"
    updated_file_as_string += "\t"
    updated_file_as_string += str(b)
    updated_file_as_string += "\n"

    updated_file_as_string += "tx"
    updated_file_as_string += "\t"
    updated_file_as_string += str(tx)
    updated_file_as_string += "\n"


    updated_file_as_string += "c"
    updated_file_as_string += "\t"
    updated_file_as_string += str(c)
    updated_file_as_string += "\n"

    updated_file_as_string += "d"
    updated_file_as_string += "\t"
    updated_file_as_string += str(d)
    updated_file_as_string += "\n"

    updated_file_as_string += "ty"
    updated_file_as_string += "\t"
    updated_file_as_string += str(ty)
    updated_file_as_string += "\n"

    sigma_x, sigma_y = sigmas

    updated_file_as_string += "sigma_x"
    updated_file_as_string += "\t"
    updated_file_as_string += str(sigma_x)
    updated_file_as_string += "\n"

    updated_file_as_string += "sigma_y"
    updated_file_as_string += "\t"
    updated_file_as_string += str(sigma_y)
    updated_file_as_string += "\n"

    static_center_a, static_center_b = static_centers
    static_center_a_x, static_center_a_y = static_center_a
    static_center_b_x, static_center_b_y = static_center_b

    updated_file_as_string += "static_center_a_x"
    updated_file_as_string += "\t"
    updated_file_as_string += str(static_center_a_x)
    updated_file_as_string += "\n"

    updated_file_as_string += "static_center_a_y"
    updated_file_as_string += "\t"
    updated_file_as_string += str(static_center_a_y)
    updated_file_as_string += "\n"

    updated_file_as_string += "static_center_b_x"
    updated_file_as_string += "\t"
    updated_file_as_string += str(static_center_b_x)
    updated_file_as_string += "\n"

    updated_file_as_string += "static_center_b_y"
    updated_file_as_string += "\t"
    updated_file_as_string += str(static_center_b_y)
    updated_file_as_string += "\n"


    a2 = warp_matrix_2[0][0]
    b2 = warp_matrix_2[0][1]
    tx2 = warp_matrix_2[0][2]
    c2 = warp_matrix_2[1][0]
    d2 = warp_matrix_2[1][1]
    ty2 = warp_matrix_2[1][2]


    updated_file_as_string += "a2"
    updated_file_as_string += "\t"
    updated_file_as_string += str(a2)
    updated_file_as_string += "\n"

    updated_file_as_string += "b2"
    updated_file_as_string += "\t"
    updated_file_as_string += str(b2)
    updated_file_as_string += "\n"

    updated_file_as_string += "tx2"
    updated_file_as_string += "\t"
    updated_file_as_string += str(tx2)
    updated_file_as_string += "\n"


    updated_file_as_string += "c2"
    updated_file_as_string += "\t"
    updated_file_as_string += str(c2)
    updated_file_as_string += "\n"

    updated_file_as_string += "d2"
    updated_file_as_string += "\t"
    updated_file_as_string += str(d2)
    updated_file_as_string += "\n"

    updated_file_as_string += "ty2"
    updated_file_as_string += "\t"
    updated_file_as_string += str(ty2)
    updated_file_as_string += "\n"

    params_file = open('stream_configuration.txt', 'w+')
    params_file.write(updated_file_as_string)
    params_file.close()
    os.chdir(current_working_directory)
    return
