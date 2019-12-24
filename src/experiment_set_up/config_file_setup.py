from . import update_camera_configuration as ucc
import os

# Fix ability to update config file later
def assign_config_files(parameter_dictionary, args, camera_configurations_folder):
    camera_a_configuration = os.path.join(camera_configurations_folder, "cam_a_default.pfs")
    camera_b_configuration = os.path.join(camera_configurations_folder, "cam_b_default.pfs")

 # Case when both Cameras read the same alternate camera configuration file: CCF*
    if args["camera_configuration_file"] is not None \
            and args["camera_configuration_file_a"] is None \
            and args["camera_configuration_file_b"] is None:
        camera_a_configuration = args["camera_configuration_file"]
        camera_b_configuration = args["camera_configuration_file"]

    # Case when Camera A reads an default camera configuration file while B Reads Alternate CCF: CCF_B*
    elif args["camera_configuration_file"] is None \
            and args["camera_configuration_file_a"] is None \
            and args["camera_configuration_file_b"] is not None:
        camera_a_configuration = os.path.join(camera_configurations_folder, "cam_a_default.pfs")
        camera_b_configuration = args["camera_configuration_file_b"]

    config_files_by_cam = {"a": camera_a_configuration, "b": camera_b_configuration}

    return config_files_by_cam

