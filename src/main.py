"""Driver class - you should be able to run main from the command line along with the some command line arguments of
  your choice, and this script will call all other modules/scripts and feed those args/params as needed."""

import argparse
import os
import stream_cameras_and_histograms as streams
import update_camera_configuration as ucc

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


parser.add_argument('-va', '--video_a', default=None,
                    help='Path to video file for Camera A (if not using camera)')
parser.add_argument('-vb', '--video_b', default=None,
                    help='Path to video file for Camera B (if not using camera)')
parser.add_argument('-ccf', '--camera_configuration_file', default=None,
                    help='Desired camera configuration file (if other than default_camera_configuration.pfs)')
parser.add_argument('-ccfa', '--camera_configuration_file_a', default=None,
                    help='Desired camera configuration file for A (if other than default_camera_configuration.pfs)')
parser.add_argument('-ccfb', '--camera_configuration_file_b', default=None,
                    help='Desired camera configuration file for B (if other than default_camera_configuration.pfs)')
parser.add_argument('-e', '--ExposureTime', type=str, default=None,
                    help='Exposure time in us (microseconds) [overwrites value in default_camera_configuration.pfs]')
parser.add_argument('-r', '--AcquisitionFrameRate', type=str, default=None,
                    help='Acquisition frame rate (Hz) [overwrites value in default_camera_configuration.pfs]')
parser.add_argument('-si', '--SaveImages', type=int, default=0,
                    help="1 if you'd like to save the images from this run, else 0.")
parser.add_argument('-sv', '--SaveVideos', type=int, default=0,
                    help="1 if you'd like to save the videos from this run, else 0.")
parser.add_argument('-fb', '--FrameBreak', type=int, default=-1,
                    help="How many frames you'd like to grab/save.")
parser.add_argument('-dhc', '--DisplayHistocam', type=int, default=0,
                    help="How many frames you'd like to grab/save.")
parser.add_argument('-shc', '--SaveHistocam', type=int, default=0,
                    help="How many frames you'd like to grab/save.")
parser.add_argument('-raw', '--Raw', type=int, default=1,
                    help="Show raw image data.")
parser.add_argument('-rf', '--ResizeFactor', type=float, default=None,
                    help="Show raw image data.")
parser.add_argument('-g', '--Grid', type=int, default=0,
                    help="Show a grid on the images.")

if __name__ == "__main__":
    print("Starting:")
    args = vars(parser.parse_args())  # Parse user arguments into a dictionaryq
    if args["video_a"] is None and args["video_b"] is None:
        print("No file specified: Attempting video stream")
    elif args["video_a"] is None or args["video_b"] is None:
        print("To run a playback of data, you must specify two valid video files.")
    else:
        validate_video_file(args["video_a"])  # Raises error if not a valid video
        validate_video_file(args["video_b"])  # Raises error if not a valid video

    relevant_parameters = ["ExposureTime", "AcquisitionFrameRate"]
    parameter_dictionary = ucc.reduce_dictionary(args, relevant_parameters)
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
        camera_a_configuration = args["camera_configuration_file_a"]
        camera_b_configuration = os.path.join(camera_configurations_folder, "default_camera_configuration.pfs")

    # Case when Camera A reads an default camera configuration file while B Reads Alternate CCF: CCF_B*
    elif args["camera_configuration_file"] is None \
            and args["camera_configuration_file_a"] is None \
            and args["camera_configuration_file_b"] is not None:
        camera_a_configuration = os.path.join(camera_configurations_folder, "default_camera_configuration.pfs")
        camera_b_configuration = args["camera_configuration_file_b"]

    else:
        camera_a_configuration = os.path.join(camera_configurations_folder, "default_camera_configuration.pfs")
        camera_b_configuration = camera_a_configuration

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
